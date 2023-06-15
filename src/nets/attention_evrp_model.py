import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple

from .graph_encoder import GraphAttentionEncoder
from .graph_decoder import GraphDecoderEVRP
from src.graph.evrp_network import EVRPNetwork
from src.utils.beam_search import CachedLookup

import os

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


class AttentionEVRPModel(nn.Module):
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
        super(AttentionEVRPModel, self).__init__()

        self.decode_type = None
        self.temp = 1.0
        self.opts = opts

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

        self.decoder = GraphDecoderEVRP(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            step_context_dim=self.step_context_dim,
            tanh_clipping=tanh_clipping,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            num_trailers=opts.num_trailers,
            num_trucks=opts.num_trucks,
            features=self.features,
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input: dict = {}, graphs: tuple = (), epoch=0, type="initial", return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        self.epoch = epoch
        self.type = type

        if len(graphs) > 0:
            self.graphs = EVRPNetwork(
                num_graphs=graphs[0].num_nodes,
                num_nodes=graphs[0].num_nodes,
                num_trucks=graphs[0].num_trucks,
                num_trailers=graphs[0].num_trailers,
                truck_names=graphs[0].truck_names,
                plot_attributes=True,
                graphs=graphs,
            )
        else:
            self.graphs = None

        if (
            self.checkpoint_encoder and self.training
        ):  # Only checkpoint if we need gradients
            embeddings = checkpoint(self._init_embed(self.encoder), input)
        else:
            embeddings = self.encoder(self._init_embed(input))

        self.encoder_data["input"] = input["coords"].cpu().detach()
        self.encoder_data["embeddings"] = embeddings.cpu().detach()

        cost, pi, decision, _log_trailers, _log_trucks, _log_nodes = self.step(input, embeddings)
        # cost, mask = self.problem.get_costs(input, pi_node)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll_trailer, ll_truck, ll_node = self._calc_log_likelihood(
            _log_trailers, _log_trucks, _log_nodes, pi
        )
        if return_pi:
            return cost, (ll_trailer + ll_truck + ll_node), pi, decision

        return cost, ll_trailer + ll_truck + ll_node  # tensor(batch_size) both

    def step(self, input, embeddings):
        outputs_trailers = []  # [(log_trailer, log_truck, log_node)]
        outputs_trucks = []
        outputs_nodes = []
        sequences_trailers = (
            []
        )  # [(from_node, to_node, truck_id, trailer_id, timestep=i)]
        sequences_trucks = []
        sequences_nodes = []
        state = self.problem.make_state(input)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        self.get_graphs()
        # Perform decoding steps
        i = 0
        while not state.all_finished():
            (trailer, truck, node), (log_trailer, log_truck, log_p) = self.decoder(
                fixed, state, temp=self.temp, decode_type=self.decode_type
            )
            state = state.update(trailer, truck, node)

            self.get_graphs(
                state=state,
                selected=(trailer, truck, node, i),
            )

            # Collect output of step
            outputs_trailers.append(log_trailer)
            outputs_trucks.append(log_truck)
            outputs_nodes.append(log_p)

            sequences_trailers.append(trailer)
            sequences_trucks.append(truck)
            sequences_nodes.append(node)

            i += 1
            
        cost = torch.where(
            state.force_stop == 1, state.lengths.sum(1)*5, state.lengths.sum(1)
        ) # TODO test other options *1.5?

        print(state.decision.shape)
        # Collected lists, return Tensor
        return (
            cost,  # (batch_size,)
            state.visited_.transpose(0,1),  # (5, batch_size, time)
            state.decision,   # (batch_size, 3, time) 3 because (selected_trailer, selected_truck, selected_node)
            torch.stack(outputs_trailers, 1),  # (batch_size, time, trailer_size)
            torch.stack(outputs_trucks, 1),  # (batch_size, time, truck_size)
            torch.stack(outputs_nodes, 1),  # (batch_size, time, graph_size)
        )

    def _calc_log_likelihood(self, _log_trailers, _log_trucks, _log_nodes, pi):
        _, node, truck, trailer, _ = pi

        # Get log_p corresponding to selected actions
        mask_trailers = trailer == -1
        index = torch.where(mask_trailers, torch.zeros_like(trailer), trailer)
        gathered_log_trailers = _log_trailers.gather(
            2, index.unsqueeze(-1).to(torch.int64)
        ).squeeze(-1)
        log_trailers = torch.where(
            mask_trailers,
            torch.zeros_like(gathered_log_trailers),
            gathered_log_trailers,
        )

        assert (
                log_trailers > -1000
        ).data.all(), "Logprobs of trailers should not be -inf, check sampling procedure!"

        mask_trucks = truck == -1
        index_truck = torch.where(mask_trucks, torch.zeros_like(truck), truck)
        gathered_log_trucks = _log_trucks.gather(
            2, index_truck.unsqueeze(-1).to(torch.int64)
        ).squeeze(-1)
        log_trucks = torch.where(
            mask_trucks, torch.zeros_like(gathered_log_trucks), gathered_log_trucks
        )

        assert (
                log_trucks > -1000
        ).data.all(), "Logprobs of trucks should not be -inf, check sampling procedure!"

        mask_nodes = node == -1
        index_node = torch.where(mask_nodes, torch.zeros_like(node), node)
        gathered_log_nodes = _log_nodes.gather(
            2, index_node.unsqueeze(-1).to(torch.int64)
        ).squeeze(-1)
        log_nodes = torch.where(
            mask_nodes, torch.zeros_like(gathered_log_nodes), gathered_log_nodes
        )

        assert (
                log_nodes > -1000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_trailers.sum(1), log_trucks.sum(1), log_nodes.sum(1)  # (batch_size,)

    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[
            :, None, :
        ]  # (batch_size, 1, embed_dim)

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

    def _init_embed(self, input):
        return self.init_embed_node(
            torch.cat((input["coords"], *(input[feat] for feat in self.features)), -1)
        )

    def _initialize_problem(self, embedding_dim: int):
        self.node_dim = 5  # x, y, avail_chargers, node_trucks, node_trailers
        self.features = ("avail_chargers", "node_trucks", "node_trailers")

        # To map input to embedding space
        self.init_embed_node = nn.Linear(self.node_dim, embedding_dim)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Embedding of truck node & trailer’s destination node
        # (In case truck and trailer are not on the same node, we use then truck and trailer location nodes)
        self.step_context_dim = 2 * embedding_dim

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings = self.encoder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def get_graphs(self, state=None, selected=[]):
        file = self.opts.save_dir + "/graphs/" + self.type + '/' + str(self.epoch)
        name = None
        if self.opts.display_graphs is not None and self.graphs is not None:
            if not os.path.exists(file):
                os.makedirs(file)

            if state is not None:
                name = "initial"
                self.graphs.clear()
                edge = self.graphs.visit_edges(tensor_to_tuples(state.visited_))
                self.graphs.update_attributes(edge)

            self.graphs.draw(
                graph_idxs=range(self.opts.display_graphs),
                selected=selected,
                with_labels=True,
                file=file,
                name=name
            )


def tensor_to_tuples(visited):
    batch_size, features, time = visited.shape
    edges = []
    for b in range(batch_size):
        batch_list = tuple(visited[b, :, time - 1].tolist())
        edges.append([batch_list])

    return edges
