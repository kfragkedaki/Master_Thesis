import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

# from src.utils.tensor_functions import compute_in_batches

from .graph_encoder import GraphAttentionEncoder
from .graph_decoder import GraphDecoder, GraphDecoderVRP
from torch.nn import DataParallel
from src.utils.beam_search import CachedLookup

# from src.utils.functions import sample_many


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """

    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key],
        )


class AttentionTSPModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        problem=None,
        n_encode_layers: int = 2,
        tanh_clipping: int = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        normalization: str = "batch",
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        opts: dict = None,
    ):
        super(AttentionTSPModel, self).__init__()

        self.decode_type = None
        self.temp = 1.0

        self.problem = problem  # env
        self.n_heads = n_heads
        self.n_encode_layers = n_encode_layers
        self.checkpoint_encoder = checkpoint_encoder
        self.encoder_data = dict()

        self._initialize_problem(embedding_dim)

        self.encoder = GraphAttentionEncoder(
            num_heads=n_heads,
            embed_dim=embedding_dim,
            num_attention_layers=n_encode_layers,
            normalization=normalization,
        )

        self.decoder = GraphDecoder(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            step_context_dim=self.step_context_dim,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            W_placeholder=self.W_placeholder,
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, graphs=None, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if (
            self.checkpoint_encoder and self.training
        ):  # Only checkpoint if we need gradients
            embeddings = checkpoint(self.encoder, self._init_embed(input))
        else:
            embeddings = self.encoder(self._init_embed(input))

        self.encoder_data["input"] = input.cpu().detach()
        self.encoder_data["embeddings"] = embeddings.cpu().detach()

        _log_p, pi = self.node_select(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        acc_log_prob = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, acc_log_prob, pi

        return cost, acc_log_prob  # tensor(batch_size) both

    def node_select(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Perform decoding steps
        i = 0
        while not state.all_finished():

            selected, log_p = self.decoder(
                fixed, state, temp=self.temp, decode_type=self.decode_type
            )
            state = state.update(selected)

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(
            sequences, 1
        )  # (batch_size, i, graph size) and (batch_size, graph size)

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (
            log_p > -1000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    # def sample_many(self, input, batch_rep=1, iter_rep=1):
    #     """
    #     :param input: (batch_size, graph_size, node_dim) input node features
    #     :return:
    #     """
    #     # Bit ugly but we need to pass the embeddings as well.
    #     # Making a tuple will not work with the problem.get_cost function
    #     return sample_many(
    #         lambda input: self._inner(*input),  # Need to unpack tuple into arguments
    #         lambda input, pi: self.problem.get_costs(
    #             input[0], pi
    #         ),  # Don't need embeddings as input to get_costs
    #         (
    #             input,
    #             self.encoder(input),
    #         ),  # Pack input with embeddings (additional input)
    #         batch_rep,
    #         iter_rep,
    #     )

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous(),
        )
        return AttentionModelFixed(
            embeddings, fixed_context, *fixed_attention_node_data
        )

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.n_heads,
                -1,
            )
            .permute(
                3, 0, 1, 2, 4
            )  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
        # Problem specific context parameters (placeholder and step context dimension)

    def _initialize_problem(self, embedding_dim: int):
        self.node_dim = 2  # x, y
        self.features = ()

        # To map input to embedding space
        self.init_embed_node = nn.Linear(self.node_dim, embedding_dim)

        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.step_context_dim = 2 * embedding_dim  # Embedding of first and last node

        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(
            -1, 1
        )  # Placeholder should be in range of activations

    def _init_embed(self, input):
        return self.init_embed_node(input)

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings = self.encoder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    # def propose_expansions(
    #         self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096
    # ):
    #     # First dim = batch_size * cur_beam_size
    #     log_p_topk, ind_topk = compute_in_batches(
    #         lambda b: self._get_log_p_topk(
    #             fixed[b.ids], b.state, k=expand_size, normalize=normalize
    #         ),
    #         max_calc_batch_size,
    #         beam,
    #         n=beam.size(),
    #     )
    #
    #     assert log_p_topk.size(1) == 1, "Can only have single step"
    #     # This will broadcast, calculate log_p (score) of expansions
    #     score_expand = beam.score[:, None] + log_p_topk[:, 0, :]
    #
    #     # We flatten the action as we need to filter and this cannot be done in 2d
    #     flat_action = ind_topk.view(-1)
    #     flat_score = score_expand.view(-1)
    #     flat_feas = flat_score > -1e10  # != -math.inf triggers
    #
    #     # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
    #     flat_parent = torch.arange(
    #         flat_action.size(-1), out=flat_action.new()
    #     ) // ind_topk.size(-1)
    #
    #     # Filter infeasible
    #     feas_ind_2d = torch.nonzero(flat_feas)
    #
    #     if len(feas_ind_2d) == 0:
    #         # Too bad, no feasible expansions at all :(
    #         return None, None, None
    #
    #     feas_ind = feas_ind_2d[:, 0]
    #
    #     return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]
    #
    # def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
    #     log_p, _ = self._get_log_p(fixed, state, normalize=normalize)
    #
    #     # Return topk
    #     if k is not None and k < log_p.size(-1):
    #         return log_p.topk(k, -1)
    #
    #     # Return all, note different from torch.topk this does not give error if less than k elements along dim
    #     return (
    #         log_p,
    #         torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(
    #             log_p.size(0), 1
    #         )[:, None, :],
    #     )
