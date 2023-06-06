import torch
from typing import NamedTuple
from src.utils.boolmask import mask_long2bool, mask_long_scatter


class StateEVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor

    # TODO Maybe not to use that but update via the graph networkx library? Seems more effective
    avail_chargers: torch.Tensor  # Keeps the available chargers per node
    node_trucks: torch.Tensor  # Keeps the trucks per node TODO Should I keep only the charged trucks?
    node_trailers: torch.Tensor  # Keeps the trailers per node TODO Should I keep only the charged trucks?

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    num_chargers: torch.Tensor  # Keep track of the total number of chargers in each node
    trucks_locations: torch.Tensor  # trucks location
    trucks_battery_levels: torch.Tensor  # trucks battery level

    trailers_locations: torch.Tensor  # trailers location
    trailers_destinations: torch.Tensor  # trailers' destinations
    trailers_status: torch.Tensor  # trailers' status
    trailers_start_time: torch.Tensor  # trailers' start_time
    trailers_end_time: torch.Tensor  # trailers' end_time

    # State
    from_loc: torch.Tensor
    to_loc: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        pass
        # if self.visited_.dtype == torch.uint8:
        #     return self.visited_
        # else:
        #     return mask_long2bool(self.visited_, n=self.coords.size(-2))

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
            from_loc=self.from_loc[key],
            to_loc=self.to_loc[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        coords = input["coords"]
        node_trucks = input["node_trucks"]
        node_trailers = input["node_trailers"]

        batch_size, num_nodes, _ = coords.size()
        _, num_trucks, _ = node_trucks.size()
        _, num_trailers, _ = node_trailers.size()

        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=coords.device)

        return StateEVRP(
            coords=coords,
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
            from_loc=prev_a,
            to_loc=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, num_nodes, dtype=torch.uint8, device=coords.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(
                    batch_size,
                    1,
                    (num_nodes + 63) // 64,
                    dtype=torch.int64,
                    device=coords.device,
                )  # Ceil
            ),
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
        # TODO select truck, trailer, node and update

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.coords[self.ids, prev_a]
        lengths = self.lengths
        if (
            self.cur_coord is not None
        ):  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(
                p=2, dim=-1
            )  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step,
        # in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(
            from_loc=first_a,
            to_loc=prev_a,
            visited_=visited_,
            lengths=lengths,
            cur_coord=cur_coord,
            i=self.i + 1,
        )

    def all_finished(self):
        # If all trailers are on their destination nodes
        pass

    def get_current_node(self):
        return self.to_loc

    def get_mask(self):
        # Mask the nodes that the truck cannot go to because of its battery limits
        # Mask the nodes on Pending mode
        pass

    def construct_solutions(self, actions):
        return actions
