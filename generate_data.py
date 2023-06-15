import argparse
import os
import numpy as np
from src.utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {4: 10.0, 10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}
    return list(
        zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            np.random.uniform(
                size=(dataset_size, vrp_size, 2)
            ).tolist(),  # Node locations
            np.random.randint(
                1, 10, size=(dataset_size, vrp_size)
            ).tolist(),  # Demand, uniform integer 1 ... 9
            np.full(
                dataset_size, CAPACITIES[vrp_size]
            ).tolist(),  # Capacity, same for whole dataset
        )
    )


def generate_evrp_data(dataset_size, graph_size, num_trailers, num_trucks):
    coords = np.random.uniform(size=(dataset_size, graph_size, 2)).tolist()
    node_chargers = np.random.randint(
        low=1, high=10, size=(dataset_size, graph_size)
    ).tolist()
    trucks_locations = [
        np.random.choice(graph_size, num_trucks, replace=False).tolist()
        for _ in range(dataset_size)
    ]
    trucks_battery_levels = np.ones(shape=(dataset_size, num_trucks)).tolist()
    trailers_locations = [
        np.random.choice(graph_size, num_trailers, replace=False).tolist()
        for _ in range(dataset_size)
    ]
    destinations = [
        [
            np.random.choice([n for n in range(graph_size) if n != node_id]).tolist()
            for node_id in trailer_origins
        ]
        for trailer_origins in trailers_locations
    ]
    start_time = np.random.randint(
        low=8.00, high=18.00, size=(dataset_size, num_trailers)
    )
    end_time = (
        start_time
        + np.round(
            np.random.uniform(low=0.5, high=2, size=(dataset_size, num_trailers)) * 2
        )
        / 2
    )

    return list(
        zip(
            coords,
            node_chargers,
            trucks_locations,
            trucks_battery_levels,
            trailers_locations,
            destinations,
            start_time.tolist(),
            end_time.tolist(),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", help="Filename of the dataset to create (ignores datadir)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Create datasets in data_dir/problem (default 'data')",
    )
    parser.add_argument(
        "--name", type=str, default="new2", help="Name to identify dataset"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem, 'tsp', 'vrp', 'evro' or 'all' to generate all",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="all",
        help="Distributions to generate for problem, default 'all'.",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=1000, help="Size of the dataset"
    )
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[4, 20],
        help="Sizes of problem instances (default 20, 50, 100)",
    )
    parser.add_argument(
        "--num_trucks", type=int, nargs="+", default=2, help="Size of the fleet"
    )
    parser.add_argument(
        "--num_trailers",
        type=int,
        nargs="+",
        default=3,
        help="Size of the deliverables",
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (
        len(opts.problems) == 1 and len(opts.graph_sizes) == 1
    ), "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        "tsp": [None],
        "vrp": [None],
        "evrp": [None],
    }

    if opts.problem == "all":
        problems = distributions_per_problem

    for problem, distributions in problems.items():
        for graph_size in opts.graph_sizes:
            datadir = os.path.join(opts.data_dir, problem)
            os.makedirs(datadir, exist_ok=True)

            if opts.filename is None:
                filename = os.path.join(
                    datadir,
                    "{}{}_{}_seed{}.pkl".format(
                        problem, graph_size, opts.name, opts.seed
                    ),
                )
            else:
                filename = check_extension(opts.filename)

            assert opts.f or not os.path.isfile(
                check_extension(filename)
            ), "File already exists! Try running with -f option to overwrite."

            np.random.seed(opts.seed)
            if problem == "tsp":
                dataset = generate_tsp_data(opts.dataset_size, graph_size)
            elif problem == "vrp":
                dataset = generate_vrp_data(opts.dataset_size, graph_size)
            elif problem == "evrp":
                dataset = generate_evrp_data(
                    opts.dataset_size, graph_size, opts.num_trailers, opts.num_trucks
                )
            else:
                assert False, "Unknown problem: {}".format(problem)

            print(dataset[0])

            save_dataset(dataset, filename)
