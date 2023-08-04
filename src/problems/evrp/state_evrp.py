import torch
from typing import NamedTuple
from src.utils.boolmask import mask_long2bool, mask_long_scatter


class StateEVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor
    distances: torch.Tensor
    num_chargers: torch.Tensor  # Keep track of the total number of chargers in each node
    trailers_destinations: torch.Tensor  # trailers' destinations
    trailers_start_time: torch.Tensor  # trailers' start_time
    trailers_end_time: torch.Tensor  # trailers' end_time

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    avail_chargers: torch.Tensor  # Keeps the available chargers per node
    node_trucks: torch.Tensor  # Keeps the trucks per node
    node_trailers: torch.Tensor  # Keeps the trailers per node
    node_charged_trucks: torch.Tensor  # Keeps the charged trucks per node

    trucks_locations: torch.Tensor  # trucks location
    trucks_battery_levels: torch.Tensor  # trucks battery level

    trailers_locations: torch.Tensor  # trailers location

    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    penalty: torch.Tensor
    reward: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    force_stop: torch.Tensor
    r_threshold: torch.Tensor

    decision: torch.Tensor
    timestep: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(
            key, slice
        )  # If tensor, idx all tensors by this tensor:

        return self._replace(
            avail_chargers=self.avail_chargers[key],
            node_trucks=self.node_trucks[key],
            node_trailers=self.node_trailers[key],
            ids=self.ids[key],
            num_chargers=self.num_chargers[key],
            trucks_locations=self.trucks_locations[key],
            trucks_battery_levels=self.trucks_battery_levels[key],
            trailers_locations=self.trailers_locations[key],
            trailers_destinations=self.trailers_destinations[key],
            trailers_start_time=self.trailers_start_time[key],
            trailers_end_time=self.trailers_end_time[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            decision=self.decision[key],
            timestep=self.timestep[key],
            penalty=self.penalty[key],
            reward=self.reward[key]
        )

    @staticmethod
    def initialize(input, r_threshold, visited_dtype=torch.uint8):
        device = input["coords"].device

        batch_size, num_nodes, coords_size = input["coords"].size()
        _, num_trucks, _ = input["trucks_locations"].size()

        return StateEVRP(
            coords=input["coords"],
            distances=(
                input["coords"][:, :, None, :] - input["coords"][:, None, :, :]
            ).norm(p=2, dim=-1),
            avail_chargers=input["avail_chargers"].to(device),
            node_trucks=input["node_trucks"].to(device),
            node_trailers=input["node_trailers"].to(device),
            node_charged_trucks=input["node_trucks"].to(device),
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[
                :, None
            ],  # Add steps dimension
            num_chargers=input["num_chargers"].to(device),
            trucks_locations=input["trucks_locations"].to(device),
            trucks_battery_levels=input["trucks_battery_levels"].to(device),
            trailers_locations=input["trailers_locations"].to(device),
            trailers_destinations=input["trailers_destinations"].to(device),
            trailers_start_time=input["trailers_start_time"].to(device),
            trailers_end_time=input["trailers_end_time"].to(device),
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=torch.zeros(
                batch_size, 5, 1, dtype=torch.uint8, device=device
            ),  # Visited as mask is easier to understand, as long more memory efficient
            lengths=torch.zeros(batch_size, num_trucks, device=device),
            cur_coord=None,
            i=torch.zeros(
                1, dtype=torch.int64, device=device
            ),  # Vector with length num_steps
            force_stop=torch.zeros(batch_size, dtype=torch.int64, device=device),
            decision=torch.zeros(batch_size, 3, 1, dtype=torch.uint8, device=device),
            timestep=torch.zeros(batch_size, 1, dtype=torch.uint8, device=device),
            penalty=torch.zeros(batch_size, 1, dtype=torch.float64, device=device),
            reward=torch.zeros(batch_size, 1, dtype=torch.float64, device=device),
            r_threshold=torch.tensor(r_threshold, dtype=torch.float64, device=device),
        )

    def get_final_cost(self):
        # to reduce the cost function we substract the reward
        cost = self.lengths.sum(1).unsqueeze(-1) + self.penalty - self.reward
        return cost.squeeze(-1)

    def update(self, selected_trailer, selected_truck, selected_node):
        # TODO Check different scenarios
        device = self.coords.device
        decision = torch.cat(
            (
                selected_trailer.unsqueeze(-1),
                selected_truck.unsqueeze(-1),
                selected_node.unsqueeze(-1),
            ),
            dim=1,
        ).unsqueeze(-1)
        # Calculate to_node
        from_node = self.trucks_locations[self.ids, selected_truck[self.ids]].squeeze(
            -1
        )  # (batch_size, 1)
        valid_truck = (selected_truck != -1).unsqueeze(-1)
        from_node = torch.where(
            valid_truck,
            from_node,
            selected_node[self.ids].to(from_node.dtype),
        )
        trailer_node_ids = self.trailers_locations[
            self.ids, selected_trailer[self.ids]
        ].squeeze(
            -1
        )  # (batch_size, 1)
        valid_trailer = (selected_trailer != -1).unsqueeze(-1)
        trailer_node_ids = torch.where(
            torch.logical_and(valid_trailer, valid_truck),
            trailer_node_ids,
            torch.tensor(-2, device=device, dtype=trailer_node_ids.dtype),
        )

        # check if trailer and truck are on the same node.
        # In case we have not selected a truck or a trailer, this condition will be False.
        condition = torch.eq(from_node, trailer_node_ids)

        avail_chargers = (self.num_chargers > 0).float()
        trucks_locations = self.trucks_locations.clone()
        trailers_locations = self.trailers_locations.clone()

        node_trucks = torch.zeros(
            size=self.node_trucks.shape, device=device
        )  # reset values
        node_charged_trucks = torch.zeros(
            size=self.node_trucks.shape, device=device
        )
        node_trailers = torch.zeros(
            size=self.node_trailers.shape, device=device
        )  # reset values

        # Update the battery level tensor: if charger is available at truck's node, set battery level to 1, else keep the same
        trucks_battery_levels = torch.where(
            self.trucks_battery_levels == 0,
            torch.tensor(1, device=device, dtype=self.trucks_battery_levels.dtype),
            self.trucks_battery_levels,
        )

        # update truck state
        selected_truck_mask = (
            selected_truck != -1
        )  # Mask indicating valid truck indices
        valid_truck_indices = selected_truck[selected_truck_mask].unsqueeze(-1)
        valid_ids = self.ids[selected_truck_mask]

        trucks_locations[valid_ids, valid_truck_indices] = (
            selected_node[selected_truck_mask]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .to(torch.float32)
        )

        trucks_battery_levels[valid_ids, valid_truck_indices] = 0
        # update trailer state
        selected_trailer_mask = (
            selected_trailer != -1
        )  # Mask indicating valid truck indices
        valid_trailer_indices = selected_trailer[selected_trailer_mask].unsqueeze(-1)
        valid_trailer_ids = self.ids[selected_trailer_mask]

        trailers_locations[valid_trailer_ids, valid_trailer_indices] = torch.where(
            condition[selected_trailer_mask].unsqueeze(
                -1
            ),  # if not valid truck this will be false
            selected_node[selected_trailer_mask]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .to(self.trailers_locations.dtype),
            self.trailers_locations[valid_trailer_ids, valid_trailer_indices],
        )

        # update features
        truck_node_location = (
            trucks_locations[valid_ids, valid_truck_indices].squeeze(-1).to(torch.int)
        )
        avail_chargers[valid_ids, truck_node_location] = torch.where(
            torch.logical_or(
                (self.num_chargers[valid_ids, truck_node_location] - 1 > 0),
                selected_trailer_mask[valid_ids].unsqueeze(-1),
            ),
            torch.tensor(1.0, device=device),
            torch.tensor(0.0, device=device),
        )
        node_trucks[
            self.ids.unsqueeze(-1).expand(trucks_locations.shape),
            trucks_locations.to(torch.int),
        ] = 1
        node_trailers[
            self.ids.unsqueeze(-1).expand(trailers_locations.shape),
            trailers_locations.to(torch.int),
        ] = 1

        # Locations of charged trucks
        locations_trucks_charged = trucks_locations[trucks_battery_levels.bool()].unsqueeze(-1).unsqueeze(-1)

        # Update node_trucks for only charged trucks
        node_charged_trucks[
            self.ids.unsqueeze(-1).expand(locations_trucks_charged.shape),
            locations_trucks_charged.to(torch.int),
        ] = 1

        # Add the length
        # self includes the previous state (not yet updated)
        cur_coord = self.coords.gather(
            1, trucks_locations.to(torch.int64).expand(-1, -1, self.coords.shape[2])
        )

        if (
            self.cur_coord is not None
        ):  # Don't add length for first action (selection of start node)
            prev_coord = self.cur_coord
        else:
            prev_coord = self.coords.gather(
                1,
                self.trucks_locations.to(torch.int64).expand(
                    -1, -1, self.coords.shape[2]
                ),
            )

        lengths = self.lengths + (cur_coord - prev_coord).norm(
            p=2, dim=-1
        )  # (batch_dim, 1)

        trailers_finished = torch.eq(
            self.trailers_destinations, trailers_locations
        ).squeeze(-1)  # when a trailer reached its destination in this or previous steps

        # penalty when the selected truck stays on the same location as before
        # while the selected trailer is still to be served
        penalty = torch.where(
            torch.logical_and(~trailers_finished[self.ids, selected_trailer.unsqueeze(-1)], (lengths == self.lengths)[self.ids, selected_truck.unsqueeze(-1)]),
            self.penalty + self.r_threshold - 0.1,
            self.penalty,
        )

        # reward when a trailer reaches its destination in this step
        # Encourage using the trucks with less distance travelled - length of the truck
        reward = torch.where(
            torch.logical_and(trailers_finished[self.ids, selected_trailer.unsqueeze(-1)],
                              ~(lengths == self.lengths)[self.ids, selected_truck.unsqueeze(-1)]),
            self.reward + 1 / (lengths[self.ids, selected_truck.unsqueeze(-1)] + int(self.i)/10),
            self.reward,
        )

        finished_batches = torch.all(
            torch.eq(self.trailers_destinations, self.trailers_locations), dim=1
        )

        timestep = torch.where(
            finished_batches,  # if it has finished before this step
            self.timestep,
            torch.full_like(selected_trailer[self.ids], int(self.i))
        )  # (batch_size, 1)
        trailer_id = torch.where(
            torch.logical_and(condition, valid_truck),
            selected_trailer[self.ids],
            torch.full_like(selected_trailer[self.ids], -1),
        )

        visited_ = torch.stack(
            (
                from_node,
                selected_node[self.ids],
                selected_truck[self.ids],
                trailer_id,
                timestep,
            )
        ).transpose(1, 0)

        if self.visited_.shape[-1] != int(self.i):
            visited_ = visited_
            decision = decision
        else:
            visited_ = torch.cat((self.visited_, visited_), dim=-1)
            decision = torch.cat((self.decision, decision), dim=-1)

        return self._replace(
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            i=self.i + 1,
            avail_chargers=avail_chargers,
            node_charged_trucks=node_charged_trucks,
            node_trucks=node_trucks,
            node_trailers=node_trailers,
            trucks_locations=trucks_locations,
            trucks_battery_levels=trucks_battery_levels,
            trailers_locations=trailers_locations,
            decision=decision,
            timestep=timestep,
            penalty=penalty,
            reward=reward,
        )

    def all_finished(self):
        # If all trailers are on their destination nodes
        _, graph_size, _ = self.num_chargers.shape
        _, num_trailers, _ = self.trailers_locations.shape
        _, num_trucks, _ = self.trucks_locations.shape

        finished = torch.eq(self.trailers_locations, self.trailers_destinations)
        finished_batches = torch.all(finished, dim=1)
        if (
            self.i > (graph_size**num_trailers) / num_trucks
        ):  # TODO Terminate when running for long (check condition)
            print(
                f"Finished Batches {torch.sum(finished_batches).item()}/{finished_batches.shape[0]}"
            )

            force_stop = self.force_stop
            force_stop[
                ~finished_batches.squeeze(-1)
            ] = 1  # all unfinished batches will be set to 1
            self._replace(force_stop=force_stop)
            return True

        return torch.all(finished)

    def get_nn(self):
        num_nodes = self.distances.shape[2]
        return (self.distances[self.ids, :, :]).topk(
            k=int(num_nodes / 2), dim=-1, largest=False
        )[1]

    def get_mask(self, selected_truck):
        # Mask current node (the cost of staying on the same node remains 0, so it is the best choice)
        # Mask the nodes that the truck cannot go to because of its battery limits
        # graph_size = self.node_trailers.shape[1]
        # device = self.node_trailers.device
        cur_nodes = (
            self.trucks_locations[self.ids, selected_truck[self.ids]]
            .squeeze(-1)
            .to(torch.int64)
        )  # if truck is -1, this will give the last value of the tensor.

        # init_mask = torch.zeros(self.num_chargers.shape, device=device)
        #
        # # mask finished batches
        # condition = torch.all(
        #     torch.eq(self.trailers_destinations, self.trailers_locations), dim=1
        # )

        # Not moving may be a good choice as well, but we need to add a penatly for that
        # so we removed the masking of the current node

        # # if batch is not yet done, then mask the current node
        # mask = torch.full_like(init_mask, False, dtype=torch.bool, device=device)
        # # mask[self.ids, cur_nodes] = True

        mask_distanced_nodes = (
            self.distances[self.ids, cur_nodes] > self.r_threshold
        )  # batch_size, 1, graph_size

        # # mask all other nodes apart from truck's location if batch has finished
        # masking_total = torch.logical_or(mask, mask_distanced_nodes)
        # output = torch.where(
        #     torch.logical_or(
        #         condition.unsqueeze(-1),
        #         torch.all(masking_total, dim=1)
        #         .unsqueeze(-1)
        #         .expand(-1, graph_size, -1),
        #     ),
        #     ~mask,
        #     masking_total,
        # )

        # return output.transpose(1, 2)  # batch_size, 1, graph_size
        return mask_distanced_nodes # batch_size, 1, graph_size
    
    def construct_solutions(self, actions):
        return actions
