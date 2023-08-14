from src.problems import EVRP
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx


def calculate_distances(input: torch.Tensor):
    distances = (input[:, None, :] - input[None, :, :]).norm(p=2, dim=-1)

    return distances


def find_shortest_path(G, start_node, end_node):
    shortest_path = nx.shortest_path(
        G, source=start_node, target=end_node, weight="weight"
    )
    return shortest_path


def select_trailer(
    trucks_locations,
    trailers_locations,
    trailers_destinations,
    distances,
    truck_usage_counts,
):
    for step in range(distances.size(1)):
        for trailer_index in range(trailers_locations.size(0)):
            trailer_loc = trailers_locations[trailer_index].item()
            if trailer_loc == trailers_destinations[trailer_index].item():
                continue

            sorted_trucks = sorted(
                range(trucks_locations.size(0)), key=lambda x: truck_usage_counts[x]
            )
            for truck_index in sorted_trucks:
                truck_loc = trucks_locations[truck_index].item()
                if distances[truck_loc, trailer_loc].item() <= step * 0.6:
                    truck_usage_counts[truck_index] += 1  # Update truck usage count
                    return trailer_index, truck_index

    raise ValueError("No feasible trailer found.")


def route_distance(route, distances):
    cost = sum(distances[route[i], route[i + 1]] for i in range(len(route) - 1))
    return cost


def select_route(graph, truck_loc, trailer_loc, trailer_destination, trailer_idx):
    route = []
    trailer_route = []
    if truck_loc != trailer_loc:
        route = (
            route
            + find_shortest_path(graph, start_node=truck_loc, end_node=trailer_loc)[:-1]
        )
        trailer_route = [-1 for _ in range(len(route))]

    tour = find_shortest_path(
        graph, start_node=trailer_loc, end_node=trailer_destination
    )
    route = route + tour
    trailer_route = trailer_route + [trailer_idx for _ in range(len(tour))]

    return route, trailer_route


def solve_evrp(graph, environment):
    # Extract information from the environment
    coords = environment["coords"]
    trucks_locations = environment["trucks_locations"].to(torch.int)
    trailers_locations = environment["trailers_locations"].to(torch.int)
    trailers_destinations = environment["trailers_destinations"].to(torch.int)

    # Calculate the distances between all pairs of nodes
    distances = calculate_distances(coords)
    sequences = []

    trucks_last_used = {
        i: -1 for i in range(trucks_locations.numel())
    }  # Last timestep each truck was used

    # Initialize truck usage counts
    truck_usage_counts = [0] * trucks_locations.size(0)

    # Initialize cost function
    total_cost = 0

    # Process all trailers
    while not (
        (trailers_locations == trailers_destinations).sum()
        >= trailers_locations.numel()
    ):
        # Select trailer and truck
        trailer_index, truck_index = select_trailer(
            trucks_locations,
            trailers_locations,
            trailers_destinations,
            distances,
            truck_usage_counts,
        )

        # Select route
        route, trailer_route = select_route(
            graph,
            trucks_locations[truck_index].item(),
            trailers_locations[trailer_index].item(),
            trailers_destinations[trailer_index].item(),
            trailer_index,
        )

        if trucks_last_used[truck_index] == -1:
            trucks_last_used[truck_index] = sum(
                i >= 0 for i in trucks_last_used.values()
            )
        for i in range(len(route) - 1):
            source_node = route[i]
            target_node = route[i + 1]
            trailer_idx = trailer_route[i]
            sequences.append(
                [
                    source_node,
                    target_node,
                    truck_index,
                    trailer_idx,
                    trucks_last_used[truck_index],
                ]
            )
            trucks_last_used[truck_index] += 2

        # Update sequence, locations and battery levels
        trucks_locations[truck_index] = route[
            -1
        ]  # The last node in the route is the new truck location
        trailers_locations[trailer_index] = route[-1]

        total_cost += route_distance(route, distances)

    sequences = torch.tensor(sequences, dtype=torch.float)  # Adds an extra dimension
    total_time = sequences[:, -1].max().item()
    return sequences, int(total_time), total_cost.item()


if __name__ == "__main__":
    env = EVRP
    dataset = env.make_dataset(
        filename="../../instances/paper_5_nodes_swapping.json", r_threshold=0.6
    )

    # display graph
    cols = 1
    edge_count = 1
    rows = math.ceil(edge_count / cols)

    fig, axs = plt.subplots(
        rows, cols, figsize=(15, rows * 5.5)
    )  # Adjusted figure size
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    if isinstance(axs, np.ndarray) and len(axs) > 1:
        axs = axs.ravel()
    else:
        axs = np.array([axs])

    for i, graph in enumerate(dataset.sampler.graphs):
        graph.draw(ax=axs[i], with_labels=True)
        axs[i].set_aspect("equal")

    plt.show()

    # display solution
    for env, evrp_graph in zip(dataset.data, dataset.sampler.graphs):
        print(solve_evrp(evrp_graph.graph, env))
