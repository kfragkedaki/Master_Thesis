from typing import List

from src.graph.evrp_graph import EVRPGraph
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils.truck_naming import get_truck_number, get_truck_names, get_trailer_number


class EVRPNetwork:
    def __init__(
        self,
        num_graphs: int,
        num_nodes: int,
        num_trucks: int,
        num_trailers: int,
        truck_names: str = None,
        plot_attributes: bool = True,
        graphs: List[EVRPGraph] = None,
        r_threshold: float = 0.6,
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

        if graphs is None:
            self.graphs: List[EVRPGraph] = []

            # generate a graph with nn nodes, nt trailers, ntr trucks
            for i in range(num_graphs):
                self.graphs.append(
                    EVRPGraph(
                        num_nodes=num_nodes,
                        num_trailers=num_trailers,
                        num_trucks=num_trucks,
                        truck_names=truck_names,
                        plot_attributes=plot_attributes,
                        r_threshold=r_threshold,
                    )
                )
        else:
            self.graphs = graphs

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

    def draw(
        self,
        graph_idxs: np.ndarray,
        selected=[],
        with_labels: bool = False,
        file=None,
        name=None,
    ) -> any:
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
        fig, axs = plt.subplots(
            num_rows, num_columns, figsize=(7 * num_columns, 5.5 * num_rows)
        )
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        # Flatten axs to easily iterate over it
        if num_rows == 1 and num_columns == 1:
            axs = [axs]
        else:
            axs = axs.ravel()

        for col, graph_idx in enumerate(graph_idxs):
            axs[col].set_aspect("equal")
            axs[col].set_xlim([-0.7, 1.3])
            axs[col].set_ylim([-0.7, 1.3])

            if with_labels and col == len(graph_idxs) - 1:
                legend = with_labels
            else:
                legend = False
            self.graphs[graph_idx].draw(ax=axs[col], with_labels=legend)

            if len(selected) > 0:
                trailer_id, truck_id, node_id, time = selected
                txt = f"{int(time)}: trailer {int(trailer_id[graph_idx])}, truck {int(truck_id[graph_idx])}, node {int(node_id[graph_idx])}"
                axs[col].set_title(
                    txt,
                    loc="right",
                    fontdict={"fontsize": 14},
                )

        time = selected[-1] if len(selected) > 0 else 0

        if file is None:
            plt.show()
            plt.close()
        else:
            plt.savefig(f"{file}/{name or int(time)}.png", bbox_inches="tight")
            plt.close()

        # convert to plot to rgb-array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def visit_edges(self, transition_matrix: np.ndarray) -> np.array:
        """
        Visits each edges specified in the transition matrix.

        Args:
            transition_matrix (np.ndarray): Shape num_graphs x 2
                where each row is [source_node_idx, target_node_idx, truck, trailer, index].
        """
        edges = []
        for i, row in enumerate(transition_matrix):
            edges.append(self.graphs[i].visit_edges(row))

        return edges

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
        chargers = np.zeros(shape=(self.num_graphs, self.num_nodes))
        for graph_index, graph in enumerate(self.graphs):
            chargers[graph_index] = graph._node_avail_chargers

        return torch.FloatTensor(chargers)

    def get_node_trucks(self) -> dict:
        """
        Returns the state of all trucks in each graph.

        Returns:
            dict: Locations and battery levels each in shape
                (num_graphs, num_trucks, 1)
        """
        state = {}
        state["locations"] = torch.zeros(size=(self.num_graphs, self.num_trucks))
        state["battery_levels"] = torch.zeros(size=(self.num_graphs, self.num_trucks))

        for graph_index, graph in enumerate(self.graphs):
            trucks = graph._node_trucks

            # True if available charged trucks
            for node_index, truck_node in enumerate(trucks):
                if truck_node is not None:
                    for name, value in truck_node.items():
                        truck_index = get_truck_number(
                            truck=name, file=self.truck_names
                        )
                        state["locations"][graph_index, truck_index] = node_index
                        state["battery_levels"][graph_index, truck_index] = value[
                            "battery_level"
                        ]

        return state  # values

    def get_node_trailers(self) -> dict:
        """
        Returns the trailers in each graph.

        Returns:
            dict: Trailers' location and destination each in shape
                (num_graphs, num_trailers, 1)
        """
        state = {}
        state["locations"] = torch.zeros(size=(self.num_graphs, self.num_trailers))
        state["destinations"] = torch.zeros(size=(self.num_graphs, self.num_trailers))
        state["start_time"] = torch.zeros(size=(self.num_graphs, self.num_trailers))
        state["end_time"] = torch.zeros(size=(self.num_graphs, self.num_trailers))

        for graph_index, graph in enumerate(self.graphs):
            trailers = graph._node_trailers

            # True if trailer exists, and not in destination node
            for node_index, trailer_node in enumerate(trailers):
                if trailer_node is not None:
                    for name, value in trailer_node.items():
                        trailer_index = get_trailer_number(trailer=name)
                        state["locations"][graph_index, trailer_index] = node_index
                        state["destinations"][graph_index, trailer_index] = value[
                            "destination_node"
                        ]
                        state["start_time"][graph_index, trailer_index] = value[
                            "start_time"
                        ]
                        state["end_time"][graph_index, trailer_index] = value[
                            "end_time"
                        ]

        return state

    def clear(self):
        for i, graph in enumerate(self.graphs):
            graph.clear()


if __name__ == "__main__":
    G = EVRPNetwork(num_graphs=3, num_nodes=4, num_trailers=3, num_trucks=2)

    G.draw(graph_idxs=range(3), with_labels=True)
