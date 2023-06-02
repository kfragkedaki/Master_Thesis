import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple

from .graph_encoder import GraphAttentionEncoder
from .graph_decoder import GraphDecoderVRP
from src.utils.beam_search import CachedLookup


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


class AttentionVRPModel(nn.Module):
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
    ):
        super(AttentionVRPModel, self).__init__()

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

        self.decoder = GraphDecoderVRP(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            step_context_dim=self.step_context_dim,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            problem=self.problem,
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
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

        self.encoder_data["input"] = input["loc"].cpu().detach()
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
        self.node_dim = 3  # x, y, demand
        self.features = ("demand",)

        # To map input to embedding space
        self.init_embed_node = nn.Linear(self.node_dim, embedding_dim)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Embedding of last node + remaining_capacity / remaining length
        self.step_context_dim = embedding_dim + 1
        self.node_dim = 3  # x, y, demand
        self.features = ("demand",)

        # Special embedding projection for depot node
        self.init_embed_depot = nn.Linear(2, embedding_dim)

    def _init_embed(self, input):
        return torch.cat(
            (
                self.init_embed_depot(input["depot"])[:, None, :],
                self.init_embed_node(
                    torch.cat(
                        (
                            input["loc"],
                            *(input[feat][:, :, None] for feat in self.features),
                        ),
                        -1,
                    )
                ),
            ),
            1,
        )

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings = self.encoder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))
