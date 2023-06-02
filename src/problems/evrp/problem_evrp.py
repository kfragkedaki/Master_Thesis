from torch.utils.data import Dataset
import torch
import os
import pickle
from src.problems.evrp.state_evrp import StateEVRP
from src.utils.beam_search import beam_search
from src.graph.evrp_network import EVRPNetwork


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


def make_instances(num_samples, graph_size, num_trucks, num_trailers, *args):
    # TODO improve this: you already have it in a nice format
    sampler = EVRPNetwork(
        num_graphs=num_samples,
        num_nodes=graph_size,
        num_trucks=num_trucks,
        num_trailers=num_trailers,
    )

    coords = sampler.get_graph_positions()
    _, avail_chargers = sampler.get_node_avail_chargers()
    _, node_trucks = sampler.get_node_trucks()
    _, node_trailers = sampler.get_node_trailers()

    instances = []
    for i in range(coords.shape[0]):
        args = coords[i], avail_chargers[i], node_trucks[i], node_trailers[i]
        instance = make_instance(args)
        instances.append(instance)

    return instances

    return make_instance(args)


def make_instance(args):
    coords, avail_chargers, node_trucks, node_trailers, *args = args

    return {
        "coords": coords,
        "avail_chargers": avail_chargers,
        "node_trucks": node_trucks,
        "node_trailers": node_trailers,
    }


class EVRPDataset(Dataset):
    def __init__(
        self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None
    ):
        super(EVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == ".pkl"

            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.data = [
                    make_instance(args) for args in data[offset : offset + num_samples]
                ]
        else:
            self.data = make_instances(num_samples, size, 2, 3)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
