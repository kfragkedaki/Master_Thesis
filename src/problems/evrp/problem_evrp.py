from torch.utils.data import Dataset
import torch
import os
import pickle
from src.problems.evrp.state_evrp import StateEVRP
from src.utils.beam_search import beam_search
from src.utils.truck_naming import get_truck_names
from src.graph.evrp_network import EVRPNetwork
from src.graph.evrp_graph import EVRPGraph


class EVRP(object):
    NAME = "evrp"

    @staticmethod
    def make_dataset(*args, **kwargs):
        return EVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(
        input,
        beam_size,
        expand_size=None,
        compress_mask=False,
        model=None,
        max_calc_batch_size=4096,
    ):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam,
                fixed,
                expand_size,
                normalize=True,
                max_calc_batch_size=max_calc_batch_size,
            )

        state = EVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instances(
    num_samples, graph_size, num_trucks, num_trailers, truck_names, *args
):
    # TODO improve this: you already have it in a nice format

    sampler = EVRPNetwork(
        num_graphs=num_samples,
        num_nodes=graph_size,
        num_trucks=num_trucks,
        num_trailers=num_trailers,
        truck_names=truck_names,
    )

    coords = sampler.get_graph_positions()
    chargers = sampler.get_node_avail_chargers()
    trailers_state = sampler.get_node_trailers()
    trucks_state = sampler.get_node_trucks()

    instances = []
    for i in range(coords.shape[0]):
        args = (
            coords[i],
            chargers[i],
            trucks_state["locations"][i],
            trucks_state["battery_levels"][i],
            trailers_state["locations"][i],
            trailers_state["destinations"][i],
            trailers_state["status"][i],
            trailers_state["start_time"][i],
            trailers_state["end_time"][i],
        )
        instance = make_instance(args)
        instances.append(instance)

    return sampler, instances


def make_instance(args):
    (
        coords,
        chargers,
        trucks_locations,
        trucks_battery_levels,
        trailers_locations,
        trailers_destinations,
        trailers_status,
        trailer_start_time,
        trailer_end_time,
    ) = args
    num_nodes = len(coords)
    node_trucks = torch.zeros(size=(num_nodes, 1))
    node_trucks[trucks_locations.to(torch.int)] = 1

    node_trailers = torch.zeros(size=(num_nodes, 1))
    node_trailers[trailers_locations.to(torch.int)] = 1
    return {
        "coords": coords,
        "num_chargers": chargers,
        "trucks_locations": trucks_locations,
        "trucks_battery_levels": trucks_battery_levels,
        "trailers_locations": trailers_locations,
        "trailers_destinations": trailers_destinations,
        "trailers_status": trailers_status,
        "trailers_start_time": trailer_start_time,
        "trailers_end_time": trailer_end_time,
        "avail_chargers": torch.where(chargers > 0, 1.0, 0.0),
        "node_trucks": node_trucks,
        "node_trailers": node_trailers,
    }


class EVRPDataset(Dataset):
    def __init__(
        self,
        filename: str = None,
        size: int = 50,
        num_samples: int = 1000000,
        offset: int = 0,
        **kwargs
    ):
        super(EVRPDataset, self).__init__()

        num_trucks = kwargs["num_trucks"] if "num_trucks" in kwargs else 2
        num_trailers = kwargs["num_trailers"] if "num_trucks" in kwargs else 3
        truck_names = kwargs["truck_names"] if "num_trucks" in kwargs else None

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = []
                for args in data[offset : offset + num_samples]:
                    instance = make_instance(args)
                    self.data.append(make_instance(args))

                    instance["num_nodes"] = len(instance["coords"])
                    instance["num_trucks"] = len(instance["trucks_locations"])
                    instance["num_trailers"] = len(instance["trailers_locations"])
                    instance["truck_names"] = truck_names

                    assert (
                        len(get_truck_names(truck_names)) > instance["num_trucks"]
                    ), "The number of truck names does not match the number of trucks"
                    self.sampler = EVRPGraph(**instance)
        else:
            assert (
                len(get_truck_names(truck_names)) > num_trucks
            ), "The number of truck names does not match the number of trucks"
            self.sampler, self.data = make_instances(
                num_samples, size, num_trucks, num_trailers, truck_names
            )

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.sampler.graphs[idx]
