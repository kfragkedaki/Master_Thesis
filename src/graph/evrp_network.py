from typing import List

from .evrp_graph import EVRPGraph
import matplotlib.pyplot as plt
import numpy as np
import torch

class EVRPNetwork:
    def __init__(
        self,
        num_graphs: int,
        num_nodes: int,
        num_trucks: int,
        num_trailers: int,
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
        self.graphs: List[EVRPGraph] = []

        # generate a graph with nn nodes, nt trailers, ntr trucks
        for _ in range(num_graphs):
            self.graphs.append(
                EVRPGraph(
                    num_nodes, num_trailers, num_trucks, plot_attributes=plot_attributes
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

    def draw(self, graph_idxs: np.ndarray) -> None:
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

            self.graphs[graph_idx].draw(ax=ax)

        plt.show()

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
                where each row is [source_node_idx, target_node_idx].
        """
        for i, row in enumerate(transition_matrix):
            self.graphs[i].visit_edge(row[0], row[1])

    def get_graph_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node in every graph as
        an ndarray of shape (num_graphs, num_nodes, 2) sorted
        by the graph and node index.

        Returns:
            np.ndarray: Node coordinates of each graph. Shape
                (num_graphs, num_nodes, 2)
        """

        node_positions = np.zeros(shape=(len(self.graphs), self.num_nodes, 2))
        for i, graph in enumerate(self.graphs):
            node_positions[i] = graph._node_positions

        return torch.FloatTensor(node_positions)

    def get_node_avail_chargers(self) -> np.ndarray:
        """
        Returns the available chargers for each node in each graph.

        Returns:
            np.ndarray: Number of available chargers of each node in shape
                (num_graphs, num_nodes, 1)
            np.ndarray: Charger availability of each node in shape
                (num_graphs, num_nodes, 1)
        """
        chargers = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))
        for i, graph in enumerate(self.graphs):
            chargers[i] = graph._node_avail_chargers

        avail_chargers = np.where(chargers > 0, 1., 0.)
        return chargers, torch.FloatTensor(avail_chargers)  # values, bool

    def get_node_trucks(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the trucks for each node in each graph.

        Returns:
            np.ndarray: Trucks of each node in shape
                (num_graphs, num_nodes, 1)
            np.ndarray: Truck availability of each node in shape
                (num_graphs, num_nodes, 1)
        """
        trucks = np.zeros(shape=(self.num_graphs, self.num_nodes, 1), dtype=dict)
        avail_trucks = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))

        for i, graph in enumerate(self.graphs):
            trucks[i] = graph._node_trucks[:, None]

            # True if available charged trucks
            for node_index, node in enumerate(graph._node_trucks):
                if node and any(truck.get('battery_level') == 1 for truck in node.values() if truck):
                    avail_trucks[i, node_index] = 1

        return trucks, torch.FloatTensor(avail_trucks)  # values, bool

    def get_node_trailers(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the trailers for each node in each graph.

        Returns:
            np.ndarray: Trailers of each node in shape
                (num_graphs, num_nodes, 1)
            np.ndarray: Trailer availability of each node in shape
                (num_graphs, num_nodes, 1)
        """
        trailers = np.zeros(shape=(self.num_graphs, self.num_nodes, 1), dtype=dict)
        avail_trailers = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))

        for i, graph in enumerate(self.graphs):
            trailers[i] = graph._node_trailers[:, None]

            # True if trailer exists, and not in destination node and if status not pending
            for node_index, trailer_node in enumerate(graph._node_trailers):
                if trailer_node is not None:
                    for trailer in trailer_node.values():
                        if trailer['destination_node'] != node_index and trailer['status'] != 'Pending':
                            avail_trailers[i, node_index] = 1

        return trailers, torch.FloatTensor(avail_trailers)  # values, bool


if __name__ == "__main__":
    fig, ax = plt.subplots()
    G = EVRPNetwork(
        num_graphs=3, num_nodes=4, num_trailers=3, num_trucks=2, plot_attributes=True
    )

    # add edges that where visited
    edges = [
        (0, 3, "Truck 1", "Trailer B", 1),
        (0, 3, "Truck 0", None, 2),
        (3, 2, "Truck 1", "Trailer A", 3),
        (3, 2, "Truck 0", "Trailer C", 4),
    ]

    G.visit_edges(edges)

    G.draw(ax=ax, with_labels=True)
