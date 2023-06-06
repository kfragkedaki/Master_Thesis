from typing import List

from src.graph.evrp_graph import EVRPGraph
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils.truck_naming import get_truck_number, get_trailer_number


class EVRPNetwork:
    def __init__(
        self,
        num_graphs: int,
        num_nodes: int,
        num_trucks: int,
        num_trailers: int,
        truck_names: str = None,
        plot_attributes: bool = False,
    ) -> List[EVRPGraph]:
        """
        Creates num_graphs random generated fully connected
        graphs with num_nodes nodes. Node positions are
        sampled uniformly in [0, 1].

        Args:
            num_graphs (int): Number of graphs to generate.
            num_nodes (int): Number of nodes in each graph.
            num_trucks (int): Number of trucks in each graph.
            num_trailers (int): Number of trailers in each graph.

        Returns:
            List[EVRPGraph]: List of num_graphs networkx graphs
        """

        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.num_trucks = num_trucks
        self.num_trailers = num_trailers
        self.truck_names = truck_names
        self.graphs: List[EVRPGraph] = []

        # generate a graph with nn nodes, nt trailers, ntr trucks
        for i in range(num_graphs):
            self.graphs.append(
                EVRPGraph(
                    num_nodes=num_nodes, num_trailers=num_trailers, num_trucks=num_trucks, truck_names=truck_names, plot_attributes=plot_attributes
                )
            )

    def get_distance(self, graph_idx: int, node_idx_1: int, node_idx_2: int) -> float:
        """
        Calculates the euclid distance between the two nodes
        within a single graph in the VRPNetwork.

        Args:
            graph_idx (int): Index of the graph
            node_idx_1 (int): Source node
            node_idx_2 (int): Target node

        Returns:
            float: Euclid distance between the two nodes
        """
        return self.graphs[graph_idx].euclid_distance(node_idx_1, node_idx_2)

    def get_distances(self, paths) -> np.ndarray:
        """
        Calculatest the euclid distance between
        each node pair in paths.

        Args:
            paths (nd.array): Shape num_graphs x 2
                where the second dimension denotes
                [source_node, target_node].

        Returns:
            np.ndarray: Euclid distance between each
                node pair. Shape (num_graphs,)
        """
        return np.array(
            [
                self.get_distance(index, source, dest)
                for index, (source, dest) in enumerate(paths)
            ]
        )

    def draw(self, graph_idxs: np.ndarray, with_labels: bool = False) -> any:
        """
        Draw multiple graphs in a matplotlib grid.

        Args:
            graph_idxs (np.ndarray): Idxs of graphs which get drawn.
                Expected to be of shape (x, ).

        Returns:
            np.ndarray: Plot as rgb-array of shape (width, height, 3).
        """

        num_columns = min(len(graph_idxs), 3)
        num_rows = np.ceil(len(graph_idxs) / num_columns).astype(int)

        # plot each graph in a 3 x num_rows grid
        plt.clf()
        fig = plt.figure(figsize=(5 * num_columns, 5 * num_rows))

        for n, graph_idx in enumerate(graph_idxs):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            if with_labels and n == len(graph_idxs)-1:
                legend = with_labels
            else:
                legend = False
            self.graphs[graph_idx].draw(ax=ax, with_labels=legend)

        plt.show(bbox_inches='tight')

        # convert to plot to rgb-array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def visit_edges(self, transition_matrix: np.ndarray) -> None:
        """
        Visits each edges specified in the transition matrix.

        Args:
            transition_matrix (np.ndarray): Shape num_graphs x 2
                where each row is [source_node_idx, target_node_idx, truck, trailer, index].
        """
        for i, row in enumerate(transition_matrix):
            self.graphs[i].visit_edge(row)

    def get_graph_positions(self) -> torch.Tensor:
        """
        Returns the coordinates of each node in every graph as
        an ndarray of shape (num_graphs, num_nodes, 2) sorted
        by the graph and node index.

        Returns:
            torch.Tensor: Node coordinates of each graph. Shape
                (num_graphs, num_nodes, 2)
        """

        node_positions = np.zeros(shape=(len(self.graphs), self.num_nodes, 2))
        for i, graph in enumerate(self.graphs):
            node_positions[i] = graph._node_positions

        return torch.FloatTensor(node_positions)

    def get_node_avail_chargers(self) -> torch.Tensor:
        """
        Returns the total number of chargers for each node in each graph.

        Returns:
            torch.Tensor: Number of available chargers of each node in shape
                (num_graphs, num_nodes, 1)
        """
        chargers = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))
        for i, graph in enumerate(self.graphs):
            chargers[i] = graph._node_avail_chargers[:, None]

        # avail_chargers = np.where(chargers > 0, 1., 0.)
        return torch.FloatTensor(chargers)

    def get_node_trucks(self) -> dict:
        """
        Returns the state of all trucks in each graph.

        Returns:
            dict: Locations and battery levels each in shape
                (num_graphs, num_trucks, 1)
        """
        state = {}
        state["locations"] = torch.zeros(size=(self.num_graphs, self.num_trucks, 1))
        state["battery_levels"] = torch.zeros(size=(self.num_graphs, self.num_trucks, 1))

        for graph_index, graph in enumerate(self.graphs):
            trucks = graph._node_trucks

            # True if available charged trucks
            for node_index, truck_node in enumerate(trucks):
                if truck_node is not None:
                    for name, value in truck_node.items():
                        truck_index = get_truck_number(truck=name, file=self.truck_names)
                        state["locations"][graph_index, truck_index, 0] = node_index
                        state["battery_levels"][graph_index, truck_index, 0] = value["battery_level"]

        return state  # values

    def get_available_trucks(self) -> np.array:
        # TODO
        # avail_trucks = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))
        # if value['battery_level'] == 1:
        #     avail_trucks[graph_index, node_index, 0] += 1

        pass

    def get_node_trailers(self) -> dict:
        """
        Returns the trailers in each graph.

        Returns:
            dict: Trailers' location, destination and status each in shape
                (num_graphs, num_trailers, 1)
        """
        state = {}
        state["locations"] = torch.zeros(size=(self.num_graphs, self.num_trailers, 1))
        state["destinations"] = torch.zeros(size=(self.num_graphs, self.num_trailers, 1))
        state["status"] = torch.zeros(size=(self.num_graphs, self.num_trailers, 1))

        for graph_index, graph in enumerate(self.graphs):
            trailers = graph._node_trailers

            # True if trailer exists, and not in destination node and if status not pending
            for node_index, trailer_node in enumerate(trailers):
                if trailer_node is not None:
                    for name, value in trailer_node.items():
                        trailer_index = get_trailer_number(trailer=name)
                        state["locations"][graph_index, trailer_index, 0] = node_index
                        state["destinations"][graph_index, trailer_index, 0] = value["destination_node"]
                        state["status"][graph_index, trailer_index, 0] = value["status"]
                        state["start_time"][graph_index, trailer_index, 0] = value["start_time"]
                        state["end_time"][graph_index, trailer_index, 0] = value["end_time"]

        return state

    def get_available_trailers(self) -> np.array:
        # TODO
        #  avail_trailers = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))

        # if value['destination_node'] != node_index and value['status'] == 1:  # 1: 'Available'
        #     avail_trailers[graph_index, node_index, 0] += 1
        pass


if __name__ == "__main__":
    G = EVRPNetwork(
        num_graphs=3, num_nodes=4, num_trailers=3, num_trucks=2, plot_attributes=True
    )

    # add edges that where visited
    edges = [[
        (0, 3, "Truck B", "Trailer 1", 1),
        (0, 3, "Truck A", None, 2),
        (3, 2, "Truck B", "Trailer 0", 3),
        (3, 2, "Truck A", "Trailer 2", 4),
    ]]

    G.visit_edges(edges)

    G.draw(graph_idxs=range(3), with_labels=True)
