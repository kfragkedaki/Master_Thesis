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


def generate_random_point(center, radius, other_nodes, num_samples=1000, max_tries=20):
    for _ in range(max_tries):
        # Generate num_samples random distances and directions
        radii = 0.1 * np.random.rand(num_samples) + radius - 0.1
        thetas = 2 * np.pi * np.random.rand(num_samples)

        # Convert polar to cartesian
        dx = radii * np.cos(thetas)
        dy = radii * np.sin(thetas)

        # Generate potential new points
        potential_points = np.column_stack([center[0] + dx, center[1] + dy])

        # Check all points at once
        for point, radius in zip(potential_points, radii):
            if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                if (
                    all(
                        np.linalg.norm(point - node) >= radius
                        for node in other_nodes
                        if node is not None
                    )
                    and np.linalg.norm(point - center) <= radius
                ):
                    return point

    return generate_random_point(center, radius, other_nodes)


def _compute_coordinates(num_nodes, r_threshold: float):
    coordinates = dict.fromkeys(range(num_nodes))
    new_point = np.random.rand(2)
    coordinates[0] = new_point

    for node in range(1, num_nodes):
        choices = range(node)
        center_idx = np.random.choice(choices)
        center = coordinates[center_idx]
        new_point = generate_random_point(center, r_threshold, coordinates.values())
        coordinates[node] = new_point

    return list(coordinates.values())


def generate_evrp_data(dataset_size, graph_size, num_trailers, num_trucks, r_threshold):
    coords = [
        _compute_coordinates(graph_size, r_threshold) for _ in range(dataset_size)
    ]
    node_chargers = np.random.randint(
        low=1, high=10, size=(dataset_size, graph_size)
    ).tolist()
    trucks_locations = [
        np.random.choice(graph_size, num_trucks).tolist() for _ in range(dataset_size)
    ]
    trucks_battery_levels = np.ones(shape=(dataset_size, num_trucks)).tolist()
    trailers_locations = [
        np.random.choice(graph_size, num_trailers).tolist() for _ in range(dataset_size)
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
        "--name", type=str, default="new", help="Name to identify dataset"
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
        default=[4, 10],
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
    parser.add_argument(
        "--battery_limit",
        type=int,
        default=[0.6, 0.1],
        help="The distance an electric vehicle can drive without recharging.",
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
    else:
        problems = {opts.problem: [None]}

    for problem, distributions in problems.items():
        for idx, graph_size in enumerate(opts.graph_sizes):
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
                    opts.dataset_size,
                    graph_size,
                    opts.num_trailers,
                    opts.num_trucks,
                    opts.battery_limit[idx],
                )
            else:
                assert False, "Unknown problem: {}".format(problem)

            print(dataset[0])

            save_dataset(dataset, filename)
