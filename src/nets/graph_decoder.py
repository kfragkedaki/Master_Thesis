import torch
import torch.nn as nn
import math


class GraphDecoder(nn.Module):
    """
    Decoder Class to generate node prediction.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        step_context_dim: int = 256,
        tanh_clipping: int = 10.0,
        temp: int = 1,
        mask_inner: bool = True,
        mask_logits: bool = True,
        W_placeholder: torch.ParameterDict = None,
    ):
        """
        Args:
            embed_dim (int): Dimension of the input embedding.
            num_heads (int): Number of attention heads.
            step_context_dim (int): Linear Propagation of the context
            tanh_clipping (int): clipping the logits
            temp (int): temperature reduce of the learning rate
            mask_inner (bool): inner masking of the probabilities
            mask_logits (bool): mask the output probabilitiea
            W_placeholder (Parameter): initialize the final and last node
        """
        super().__init__()
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.W_placeholder = W_placeholder
        self.num_heads = num_heads
        self.temp = temp

        self.project_step_context = nn.Linear(step_context_dim, embed_dim, bias=False)
        assert embed_dim % num_heads == 0
        # Note num_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        fixed_attention: any,
        state: any,
        normalize=True,
        decode_type: str = None,
        temp: int = 1,
    ):
        """
        Forward method of the Decoder

        Args:
            node_embs (torch.Tensor): Node embeddings with shape (batch_size, num_nodes, emb_dim)
            mask (torch.Tensor, optional): Node mask with shape (batch_size, num_nodes). Defaults to None.
            load (torch.Tensor, optional): Load of the vehicle with shape (batch_size, 1). Defaults to None.
            C (int, optional): Hyperparameter to regularize logit calculation. Defaults to 10.
            rollout (bool, optional): Determines if prediction is sampled or maxed. Defaults to False.

        Returns:
            torch.Tensor: Node prediction for each graph with shape (batch_size, 1)
            torch.Tensor: Log probabilities
        """
        self.decode_type = decode_type

        # Compute query = context node embedding
        query = fixed_attention.context_node_projected + self.project_step_context(
            self._get_parallel_step_context(fixed_attention.node_embeddings, state)
        )

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(
            fixed_attention, state
        )

        # Compute the mask
        mask = state.get_mask()
        # Compute logits (unnormalized log_p)
        log_p, glimpse = self.get_attention_glimpse(
            query, glimpse_K, glimpse_V, logit_K, mask
        )

        if normalize:
            log_p = torch.log_softmax(log_p / temp, dim=-1)

        assert not torch.isnan(log_p).any()

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected = self._select_node(
            log_p.exp()[:, 0, :], mask[:, 0, :]
        )  # Squeeze out steps dimension

        return selected, log_p

    def get_attention_glimpse(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.num_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (num_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(
            batch_size, num_steps, self.num_heads, 1, key_size
        ).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (num_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(
            glimpse_Q, glimpse_K.transpose(-2, -1)
        ) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[
                mask[None, :, :, None, :].expand_as(compatibility)
            ] = -math.inf

        # Batch matrix multiplication to compute heads (num_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4)
            .contiguous()
            .view(-1, num_steps, 1, self.num_heads * val_size)
        )

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(
            -2
        ) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(
                1, selected.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        # TSP
        if (
            num_steps == 1
        ):  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(
                    batch_size, 1, self.W_placeholder.size(-1)
                )
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(
                        batch_size, 2, embeddings.size(-1)
                    ),
                ).view(batch_size, 1, -1)
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(
                batch_size, num_steps - 1, embeddings.size(-1)
            ),
        )
        return torch.cat(
            (
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(
                    batch_size, 1, self.W_placeholder.size(-1)
                ),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat(
                    (
                        embeddings_per_step[:, 0:1, :].expand(
                            batch_size, num_steps - 1, embeddings.size(-1)
                        ),
                        embeddings_per_step,
                    ),
                    2,
                ),
            ),
            1,
        )


class GraphDecoderVRP(GraphDecoder):
    """
    Decoder Class to generate node prediction.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        step_context_dim: int = 256,
        tanh_clipping: int = 10.0,
        temp: int = 1,
        mask_inner: bool = True,
        mask_logits: bool = True,
        W_placeholder: torch.ParameterDict = None,
        problem: classmethod = None,
    ):
        """
        Args:
            embed_dim (int): Dimension of the input embedding.
            num_heads (int): Number of attention heads.
            step_context_dim (int): Linear Propagation of the context
            tanh_clipping (int): clipping the logits
            temp (int): temperature reduce of the learning rate
            mask_inner (bool): inner masking of the probabilities
            mask_logits (bool): mask the output probabilitiea
            W_placeholder (Parameter): initialize the final and last node
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            step_context_dim=step_context_dim,
            tanh_clipping=tanh_clipping,
            temp=temp,
            mask_inner=mask_inner,
            mask_logits=mask_logits,
            W_placeholder=W_placeholder,
        )

        self.problem = problem

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        # Embedding of previous node + remaining capacity
        if from_depot:
            # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
            # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
            return torch.cat(
                (
                    embeddings[:, 0:1, :].expand(
                        batch_size, num_steps, embeddings.size(-1)
                    ),
                    # used capacity is 0 after visiting depot
                    self.problem.VEHICLE_CAPACITY
                    - torch.zeros_like(state.used_capacity[:, :, None]),
                ),
                -1,
            )
        else:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1)),
                    ).view(batch_size, num_steps, embeddings.size(-1)),
                    self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None],
                ),
                -1,
            )


class GraphDecoderEVRP(GraphDecoder):
    """
    Decoder Class to generate node prediction.
    """

    def __init__(self):
        super().__init__()

    def _get_attention_node_data(self, fixed, state):
        # Need to provide information of how much each node has already been served
        # Clone demands as they are needed by the backprop whereas they are updated later
        glimpse_key_step, glimpse_val_step, logit_key_step = self.project_node_step(
            state.demands_with_depot[:, :, :, None].clone()
        ).chunk(3, dim=-1)

        # Projection of concatenation is equivalent to addition of projections but this is more efficient
        return (
            fixed.glimpse_key + self._make_heads(glimpse_key_step),
            fixed.glimpse_val + self._make_heads(glimpse_val_step),
            fixed.logit_key + logit_key_step,
        )

    # def _get_parallel_step_context(self, embeddings, state, from_depot=False):
