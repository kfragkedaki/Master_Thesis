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

    trucks_locations: torch.Tensor  # trucks location
    trucks_battery_levels: torch.Tensor  # trucks battery level

    trailers_locations: torch.Tensor  # trailers location
    trailers_status: torch.Tensor  # trailers' status

    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(
            key, slice
        )  # If tensor, idx all tensors by this tensor:

        return self._replace(
            avail_chargers=self.avail_chargers[key],
            node_trucks=self.node_trucks[key],
            node_trailers=self.node_trailers[key],
            ids=self.ids[key],
            num_chargers=self.num_chargers.truck[key],
            trucks_locations=self.trucks_locations.truck[key],
            trucks_battery_levels=self.trucks_battery_levels.truck[key],
            trailers_locations=self.trailers_locations.truck[key],
            trailers_destinations=self.trailers_destinations.truck[key],
            trailers_status=self.trailers_status.truck[key],
            trailers_start_time=self.trailers_start_time[key],
            trailers_end_time=self.trailers_end_time[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        coords = input["coords"]
        node_trucks = input["node_trucks"]
        node_trailers = input["node_trailers"]

        batch_size, num_nodes, coords_size = coords.size()
        _, num_trucks, _ = input["trucks_locations"].size()

        return StateEVRP(
            coords=coords,
            distances=(coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=-1),
            avail_chargers=input["avail_chargers"],
            node_trucks=node_trucks,
            node_trailers=node_trailers,
            ids=torch.arange(batch_size, dtype=torch.int64, device=coords.device)[
                :, None
            ],  # Add steps dimension
            num_chargers=input["num_chargers"],
            trucks_locations=input["trucks_locations"],
            trucks_battery_levels=input["trucks_battery_levels"],
            trailers_locations=input["trailers_locations"],
            trailers_destinations=input["trailers_destinations"],
            trailers_status=input["trailers_status"],
            trailers_start_time=input["trailers_start_time"],
            trailers_end_time=input["trailers_end_time"],
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=torch.zeros(batch_size, 5, 1, dtype=torch.uint8, device=coords.device),  # Visited as mask is easier to understand, as long more memory efficient
            lengths=torch.zeros(batch_size, num_trucks, device=coords.device),
            cur_coord=None,
            i=torch.zeros(
                1, dtype=torch.int64, device=coords.device
            ),  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (
            self.coords[self.ids, self.trailers_destinations, :] - self.cur_coord
        ).norm(
            p=2, dim=-1
        )  # TODO fix this

    def update(self, selected):
        # TODO Check
        selected_trailer, selected_truck, selected_node = selected

        # Calculate to_node
        from_node = self.trucks_locations[self.ids, selected_truck[self.ids]].squeeze(-1)  # (batch_size, 1)
        from_node = torch.where((selected_truck != -1).unsqueeze(-1), from_node, torch.tensor(-1, device=self.trucks_locations.device))
        trailer_node_ids = self.trailers_locations[self.ids, selected_trailer[self.ids]].squeeze(-1)  # (batch_size, 1)
        condition = torch.eq(from_node, trailer_node_ids)

        avail_chargers = (self.num_chargers > 0).float()
        trucks_locations = self.trucks_locations.clone()
        trailers_locations = self.trailers_locations.clone()
        # trailers_status = torch.ones(size=self.trailers_status.shape)  # reset values
        node_trucks = torch.zeros(size=self.node_trucks.shape)  # reset values
        node_trailers = torch.zeros(size=self.node_trailers.shape)  # reset values
        # trucks_battery_levels = torch.ones(size=self.trucks_battery_levels.shape)  # reset values
        # Update the battery level tensor: if charger is available at truck's node, set battery level to 1, else keep the same
        # TODO already done, check with Jonas.
        charger_avail_trucks = avail_chargers.gather(1, self.trucks_locations.to(torch.int64)) # (batch_size, num_trucks, 1)
        trucks_battery_levels = torch.where((self.trucks_battery_levels == 0) & (charger_avail_trucks == 1), torch.tensor(1, device=self.trucks_locations.device), self.trucks_battery_levels)

        # update truck state
        # trucks_locations[self.ids, selected_truck[:, None]] = torch.where(condition.unsqueeze(-1), selected_node[:, None, None].to(torch.float32), self.trailers_locations[self.ids, selected_trailer[:, None]])
        selected_truck_mask = (selected_truck != -1)  # Mask indicating valid truck indices
        valid_truck_indices = selected_truck[selected_truck_mask].unsqueeze(-1)
        valid_ids = self.ids[selected_truck_mask]
        trucks_locations[valid_ids, valid_truck_indices] = selected_node[selected_truck_mask].unsqueeze(-1).unsqueeze(-1).to(torch.float32)
        trucks_battery_levels[valid_ids, valid_truck_indices] = 0

        # update trailer state
        selected_trailer_mask = (selected_trailer != -1)  # Mask indicating valid truck indices
        valid_trailer_indices = selected_trailer[selected_trailer_mask].unsqueeze(-1)
        valid_trailer_ids = self.ids[selected_trailer_mask]
        trailers_locations[valid_trailer_ids, valid_trailer_indices] = torch.where(condition[selected_trailer_mask].unsqueeze(-1), selected_node[selected_trailer_mask].unsqueeze(-1).unsqueeze(-1).to(torch.float32), self.trailers_locations[valid_trailer_ids, valid_trailer_indices])
        # trailers_status[self.ids, selected_trailer[self.ids]] = torch.where(condition.unsqueeze(-1), 1., 0.)  # TODO Check if needed and condition

        # update features
        selected_nodes = trucks_locations[valid_ids, valid_truck_indices].squeeze(-1).to(torch.int)
        avail_chargers[valid_ids, selected_nodes] = torch.where(self.num_chargers[valid_ids, selected_nodes]-1 > 0, 1., 0.)  # TODO
        node_trucks[self.ids.unsqueeze(-1).expand(trucks_locations.shape), trucks_locations.to(torch.int), :] = 1
        node_trailers[self.ids.unsqueeze(-1).expand(trailers_locations.shape), trailers_locations.to(torch.int), :] = 1

        # Add the length
        # node_truck = self.trucks_locations.squeeze(-1).gather(1,selected_truck[:, None])
        cur_coord = self.coords.gather(1, trucks_locations.to(torch.int64).expand(-1, -1, self.coords.shape[2]))

        if (
            self.cur_coord is not None
        ):  # Don't add length for first action (selection of start node)
            prev_coord = self.cur_coord
        else:
            prev_coord = self.coords.gather(1, self.trucks_locations.to(torch.int64).expand(-1, -1, self.coords.shape[2]))

        lengths = self.lengths + (cur_coord - prev_coord).norm(
            p=2, dim=-1
        )  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step,
        # in which case we can check this way if we should update
        # next_node = torch.where(condition.unsqueeze(-1), selected_node[:, None, None].to(torch.float32), self.trailers_locations[self.ids, selected_trailer[:, None]])[:, 0]

        timestep = torch.full_like(selected_trailer[self.ids], int(self.i))
        trailer_id = torch.where(condition, selected_trailer[self.ids], torch.full_like(selected_trailer[self.ids], -1))
        visited_ = torch.stack((from_node, selected_node[self.ids], selected_truck[self.ids], trailer_id, timestep)).transpose(1, 0)

        if self.visited_.shape[-1] != int(self.i):
            visited_ = visited_
        else:
            visited_ = torch.cat((self.visited_, visited_), dim=-1)

        return self._replace(
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            i=self.i + 1,
            avail_chargers=avail_chargers,
            node_trucks=node_trucks,
            node_trailers=node_trailers,
            trucks_locations=trucks_locations,
            trucks_battery_levels=trucks_battery_levels,
            trailers_locations=trailers_locations,
            # trailers_status=trailers_status,
         )

    def all_finished(self):
        # If all trailers are on their destination nodes
        _, graph_size, _ = self.num_chargers.shape
        _, num_trailers, _ = self.trailers_locations.shape

        if self.i > num_trailers**graph_size:  # TODO Terminate when running for long (check condition)
            print("TOO MANY RUNS")
            return True
        return torch.all(torch.eq(self.trailers_locations, self.trailers_destinations))

    def get_nn(self):
        num_nodes = self.distances.shape[2]
        return (self.distances[self.ids, :, :]).topk(k=int(num_nodes-1/2), dim=-1, largest=False)[1]

    def get_mask(self, selected_truck):
        # TODO check
        # Mask from node (the cost of staying on the same node remains 0 so it it the best choice)
        #   We should allow staying on the same node only in case no truck has been selected TODO
        # Mask the nodes that the truck cannot go to because of its battery limits
        cur_nodes = self.trucks_locations[self.ids, selected_truck[self.ids]].squeeze(-1).to(torch.int64)
        init_mask = torch.zeros(self.num_chargers.shape)
        mask = torch.full_like(init_mask, False, dtype=torch.bool)
        mask_finished = torch.full_like(init_mask, True, dtype=torch.bool)

        # # mask all apart from neighbors  # TODO but then we may have disconnected graphs, TOO MANY RUNS even with sampling
        # nns = self.get_nn()[self.ids, 0, cur_nodes, :].transpose(1, 2)  # batch_size, neighbors, 1
        # mask.scatter_(1, nns, True)
        # mask = ~mask

        condition = torch.all(torch.eq(self.trailers_destinations, self.trailers_locations), dim=1)

        # mask current node  # TODO probably not moving maybe a good choice as well, but we need to add a penatly for that?
        mask[self.ids, cur_nodes] = True
        mask_finished[self.ids, cur_nodes] = False

        output = torch.where(condition.unsqueeze(-1).expand(-1, 4, -1), mask_finished, mask)
        
        return output.transpose(1, 2)  # batch_size, 1, graph_size

    def construct_solutions(self, actions):
        return actions
