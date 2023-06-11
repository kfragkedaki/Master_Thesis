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
        self.temp = temp

        log_p, mask = self._get_log_p(fixed_attention, state, normalize)

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected = self._select_node(
            log_p.exp()[:, 0, :], mask[:, 0, :]
        )  # Squeeze out steps dimension

        return selected, log_p

    def _get_log_p(self, fixed, state, normalize=True):
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings,
                                                                          state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)  
        log_p, glimpse = self._get_attention_glimpse(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_attention_glimpse(self, query, glimpse_K, glimpse_V, logit_K, mask):

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
            W_placeholder (Parameter): None
            problem (classmethod): parameters of the problem
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
        num_trailers: int = 3,
        num_trucks: int = 2,
        feed_forward_hidden: int = 512,
        features: any = (),
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
            W_placeholder (Parameter): None
            problem (classmethod): parameters of the problem
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
        self.features = features

        # trailer decoder_extraction
        self.FF_trailer = (
            nn.Sequential(
                nn.Linear(
                    num_trailers * (embed_dim + 1), embed_dim
                ),  # trailer feature extraction
                # add 1 more value that indicates if there is at least one charged truck on this node
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )
        self.select_trailer = nn.Linear(embed_dim, num_trailers)  # vehicle selection

        # truck decoder
        self.FF_truck = (
            nn.Sequential(
                nn.Linear(num_trucks * 3, embed_dim),  # truck feature extraction
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )
        self.FF_selected_trailer = (
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )  # TODO check without this  as well
        self.select_truck = nn.Linear(embed_dim * 2, num_trucks)  # vehicle selection

        # Need to include the updated information per node in (glimpse key, glimpse value, logit key)
        self.project_node_step = nn.Linear(len(self.features), 3 * embed_dim, bias=False)

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
        self.temp = temp

        selected_trailer, log_trailer = self.trailer_select(state, fixed_attention.node_embeddings)
        selected_truck, log_truck = self.truck_select(state, fixed_attention.node_embeddings, selected_trailer)

        log_p, mask = self._get_log_p(fixed_attention, state, selected_trailer, selected_truck, normalize)

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected_node = self._select_node(
            log_p.exp()[:, 0, :], mask[:, 0, :]
        )  # Squeeze out steps dimension
        # selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :], state, truck,
        #                              sequences)  # Squeeze out steps dimension

        return (selected_trailer, selected_truck, selected_node), (log_trailer, log_truck, log_p[:, 0, :])

    def _get_log_p(self, fixed, state, trailer, truck, normalize=True):
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings,
                                                                        state, trailer, truck))  # after project: [batch_size, 1, embed_dim]

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # TODO Fix Compute the mask
        mask = state.get_mask(truck)  # [batch_size, 1, graph_size]

        # Compute logits (unnormalized log_p)  log_p:[batch_size, num_veh, graph_size], glimpse:[batch_size, num_veh, embed_dim]
        log_p, glimpse = self._get_attention_glimpse(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, trailer, truck):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param state: StateEVRP
        :return: (batch_size, num_steps, context_dim)
        """

        # Embeddings of truck node & trailer’s destination node.
        # In case truck and trailer are not on the same node,
        # we use then truck and trailer nodes.

        from_node = state.trucks_locations[state.ids, truck[:, None]].squeeze(-1)  # (batch_size, 1)
        trailer_node_ids = state.trailers_locations[state.ids, trailer[:, None]].squeeze(-1)  # (batch_size, 1)
        destination_node_ids = state.trailers_destinations[state.ids, trailer[:, None]].squeeze(-1)

        to_node = torch.where(torch.eq(from_node, trailer_node_ids), destination_node_ids, trailer_node_ids)

        return torch.cat(
            (
                embeddings[state.ids, from_node.to(torch.int64)],
                embeddings[state.ids, to_node.to(torch.int64)],
            ),
            -1,
        )

    def _get_attention_node_data(self, fixed, state):
        # Need to provide information of how much each node has already been served
        # Clone demands as they are needed by the backprop whereas they are updated later
        glimpse_key_step, glimpse_val_step, logit_key_step = self.project_node_step(
            torch.cat(
                (state.avail_chargers.clone(),
                 state.node_trucks.clone(),
                 state.node_trailers.clone()),
                2)
        )[:, None].chunk(3, dim=-1)

        # Projection of concatenation is equivalent to addition of projections but this is more efficient
        return (
            fixed.glimpse_key + self._make_heads(glimpse_key_step),
            fixed.glimpse_val + self._make_heads(glimpse_val_step),
            fixed.logit_key + logit_key_step,
        )

    def trailer_select(self, state, embeddings):
        batch_size, _, embedding_size = embeddings.size()
        embeddings_extended = torch.cat((embeddings, state.node_trucks), -1)
        trailer_embeddings = embeddings_extended.gather(
            1, state.trailers_locations.to(torch.int64).expand(-1, -1, (embedding_size+1))
        )
        trailer_context = trailer_embeddings.contiguous().reshape(batch_size, -1)
        out = self.FF_trailer(trailer_context)
        out = self.select_trailer(out)

        # masked pending trailer or trailer that have reached their destination node TODO should I include also the pending trailer?
        # TOD0 also consider that if you add the status then you need to adjust the truck_select
        # mask = torch.logical_or(torch.eq(state.trailers_destinations, state.trailers_locations),
        #     torch.eq(state.trailers_status, torch.zeros(size=state.trailers_status.shape)))
        # The purpose of the trailer status is that we do not want to bring more trailers to the node when in the previous step we have selected
        # a truck to move towards a node, but maybe we should let the algorithm learn that from the cost and mask only the unfeasible solutions

        mask = torch.eq(state.trailers_destinations, state.trailers_locations)

        out[mask.squeeze(-1)] = -math.inf
        log_trailer = torch.log_softmax(out, dim=1)  # (batch_size, num_trailer)

        probs = torch.softmax(out, dim=1)
        masking = torch.all(mask, dim=1).squeeze(-1)
        if self.decode_type == "greedy":
            _, trailer = probs.max(1)  # index of the trailer so (batch_size,)
            trailer[masking] = -1  # TODO Should I keep this and change accordingly the code, this indicates that if one batch has finished then set trailer to -1
            
            assert not mask.squeeze(-1).gather(
                1, trailer.unsqueeze(-1)
            ).data.any(), f"Decode greedy: infeasible action has maximum probability for trailer {out}"

        elif self.decode_type == "sampling":
            trailer = torch.full((batch_size,), -1)  # TODO check with Jonas
            trailer[~masking] = probs[~masking].multinomial(1).squeeze(1)
            # trailer = probs.multinomial(1).squeeze(-1) # TODO same as above, (but if all nans does not work ith multinomial)

            while mask[~masking].squeeze(-1).gather(1, trailer[~masking].unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling for trailer!")
                trailer[~masking] = probs[~masking].multinomial(
                    1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        # TODO based on the above code, we currently have nans when a batch has ended
        # assert not torch.isnan(log_trailer).any(), "Zero probabilities for truck"

        return trailer, log_trailer

    def truck_select(self, state, embeddings, trailer):
        batch_size, _, embedding_size = embeddings.size()
        # trailer_embedding = embeddings.gather(
        #     1, trailer[:, None, None].expand(-1, -1, 128)
        # ) TODO check
        trailer_embeddings = []
        for i in range(batch_size):
            if trailer[i] == -1:
                # No trailer for this batch, use a zeroed-out embedding
                trailer_embeddings.append(torch.zeros(1, embedding_size).to(embeddings.device))
            else:
                # A valid trailer exists for this batch, calculate its embedding
                trailer_embeddings.append(embeddings[i, trailer[i]].view(1, -1))

        trailer_embedding = torch.cat(trailer_embeddings, dim=0).unsqueeze(1)
        trailer_out = self.FF_selected_trailer(trailer_embedding)

        trucks_coords = state.coords.gather(
            1, state.trucks_locations.contiguous().expand(-1, -1, 2).to(torch.int64)
        ).transpose(0, 1)

        trucks_battery_levels = state.trucks_battery_levels.transpose(0, 1)
        truck_context = torch.cat(  # (batch_size, num_trucks, 3)
            (trucks_coords[0, :],
             trucks_battery_levels[0, :],
             trucks_coords[1, :],
             trucks_battery_levels[1, :],
             ), -1
        ).contiguous()[:, None, :]
        truck_out = self.FF_truck(truck_context)

        context = torch.cat((trailer_out, truck_out), -1).view(
            batch_size, embedding_size * 2
        )
        out = self.select_truck(context)

        # mask trucks that do not have battery
        mask = (state.trucks_battery_levels == 0)
        out[mask.squeeze(-1)] = -math.inf

        log_truck = torch.log_softmax(out, dim=1)  # (batch_size, num_trucks)

        probs = torch.softmax(out, dim=1)
        masking = torch.all(mask, dim=1).squeeze(-1)
        if self.decode_type == "greedy":
            _, truck = probs.max(1)  # index of the truck so (batch_size,)
            truck[masking] = -1  # TODO Should I keep this and change accordingly the code, this indicates that if one batch has finished then set trailer to -1
            assert not mask.squeeze(-1).gather(
                1, truck.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability for truck"

        elif self.decode_type == "sampling":
            # truck = torch.softmax(out, dim=1).multinomial(1).squeeze(-1)
            # while mask.squeeze(-1).gather(1, truck.unsqueeze(-1)).data.any():
            #     print("Sampled bad values, resampling for truck!")
            #     truck = truck.multinomial(1).squeeze(1)

            truck = torch.full((batch_size,), -1)  # TODO check with Jonas
            truck[~masking] = probs[~masking].multinomial(1).squeeze(1)

            while mask[~masking].squeeze(-1).gather(1, truck[~masking].unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling for truck!")
                truck[~masking] = probs[~masking].multinomial(
                    1).squeeze(1)
        else:
            assert False, "Unknown decode type"

        # truck[torch.all(mask, dim=1).squeeze(-1)] = -1  # TODO if limited chargers in a node, then all trucks are masked
        # assert not torch.isnan(log_truck).any(), "Zero probabilities for truck"

        return truck, log_truck

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"
        # selected = (state.get_current_node()).clone()
        # batch_size, _ = (state.get_current_node()).size()

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            # _, selected[torch.arange(batch_size), veh] = probs.max(1)
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

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.num_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.num_heads,
                -1,
            )
            .permute(
                3, 0, 1, 2, 4
            )  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )