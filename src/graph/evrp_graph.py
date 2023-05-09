import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import string


class EVRPGraph:
    graph: nx.Graph = nx.Graph()

    def __init__(
        self,
        num_nodes: int,
        num_trailers: int,
        num_trucks: int,
        plot_attributes: bool = False,
    ):
        """
        Creates a fully connected and directed graph with node_num nodes.
        Coordinates of each node will be sampled randomly.

        Args:
            node_num (int): Number of nodes in the graph.
        """
        self.num_nodes = num_nodes
        self.num_trucks = num_trucks
        self.num_trailers = num_trailers
        self.plot_attributes = plot_attributes

        # generate graph and set node position
        self.graph = nx.complete_graph(num_nodes, nx.MultiDiGraph())
        self.set_default_attributes()

    def set_default_attributes(self):
        """
        Sets the default colors of the nodes
        as attributes. Nodes are black except
        depots which are colored in red.

        Edges are initially marked as unvisited.
        """

        # Node Attributes

        # set coordinates for each node
        node_position = dict(enumerate(np.random.rand(self.num_nodes, 2)))
        nx.set_node_attributes(self.graph, node_position, name="coordinates")

        # set num_chargers for each node
        num_chargers = np.random.randint(low=1, high=10, size=(self.num_nodes, 1))
        nx.set_node_attributes(
            self.graph, dict(enumerate(num_chargers)), name="num_chargers"
        )

        # set available_chargers for each node
        nx.set_node_attributes(
            self.graph, dict(enumerate(num_chargers)), name="available_chargers"
        )

        # set trailers
        nx.set_node_attributes(self.graph, None, name="trailers")

        trailer_index = string.ascii_uppercase
        trailer_origin_nodes = np.random.choice(self.graph.nodes, self.num_trailers)

        for i, node_id in enumerate(trailer_origin_nodes):
            # Assign a random destination to each trailer
            destination = np.random.choice(
                [n for n in self.graph.nodes if n != node_id]
            )
            start_time = np.random.randint(low=8.00, high=18.00)
            time_frame = np.round(np.random.uniform(low=0.5, high=2) * 2) / 2

            if self.graph.nodes[node_id]["trailers"] is None:
                self.graph.nodes[node_id]["trailers"] = {}
            self.graph.nodes[node_id]["trailers"][f"Trailer {trailer_index[i]}"] = {
                "destination_node": destination,
                "start_time": np.round(start_time, 2),
                "end_time": start_time + time_frame,
            }

        # set trucks
        nx.set_node_attributes(self.graph, None, name="trucks")
        truck_nodes = np.random.choice(self.graph.nodes, self.num_trucks)

        for i, node_id in enumerate(truck_nodes):
            if self.graph.nodes[node_id]["trucks"] is None:
                self.graph.nodes[node_id]["trucks"] = {}
            self.graph.nodes[node_id]["trucks"][f"Truck {i}"] = {"battery_level": 1}

        # set general attributes
        nx.set_node_attributes(self.graph, "black", "node_color")

    def get_trailer_labels(self, data):
        node_trailers = {}
        for node_id, trailers_data in data.items():
            if trailers_data is not None:
                # origin
                if node_id not in node_trailers:
                    node_trailers[node_id] = ""
                node_trailers[node_id] += str(list(trailers_data.keys())) + "\n"

                for trailer_id, trailer_data in trailers_data.items():
                    destination_node = trailer_data["destination_node"]

                    # destination
                    if destination_node not in node_trailers:
                        node_trailers[destination_node] = ""
                    node_trailers[
                        destination_node
                    ] += f"{trailer_id} : {trailer_data['start_time']:.2f} - {trailer_data['end_time']:.2f} \n"

        return node_trailers

    def bezier_control_point(self, pos_source, pos_target, offset):
        x1, y1 = pos_source
        x2, y2 = pos_target
        x_mid = (x1 + x2) / 2
        y_mid = (y1 + y2) / 2
        x_control = x_mid - offset * (y2 - y1)
        y_control = y_mid + offset * (x2 - x1)

        return x_control, y_control

    def draw(self, ax, with_labels=False):
        """
        Draws the graph as a matplotlib plot.
        """

        # draw nodes according to color and position attribute
        pos = nx.get_node_attributes(self.graph, "coordinates")
        node_colors = nx.get_node_attributes(self.graph, "node_color").values()
        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            node_color=node_colors,
            ax=ax,
            node_size=400,
            margins=0.5,
        )

        # draw attributes
        if self.plot_attributes:
            color = {
                "Trailer A": "blue",
                "Trailer B": "red",
                "Trailer C": "green",
                None: "black",
            }
            style = {"Truck 1": "solid", "Truck 2": "dashed", None: "solid"}

            # chargers
            node_num_chargers = nx.get_node_attributes(self.graph, "num_chargers")
            node_available_chargers = nx.get_node_attributes(
                self.graph, "available_chargers"
            )

            node_num_chargers = {
                node_id: f"{avail_charg[0]} / {num_charg[0]}"
                for (node_id, num_charg), avail_charg in zip(
                    node_num_chargers.items(), node_available_chargers.values()
                )
            }
            nx.draw_networkx_labels(
                self.graph, pos=pos, labels=node_num_chargers, ax=ax, font_size=8
            )
            # trucks
            label_offset = np.array([0, 0.09])
            truck_label_pos = {k: (v - 0.6 * label_offset) for k, v in pos.items()}

            node_trucks_data = nx.get_node_attributes(self.graph, "trucks")
            node_trucks_labels = {
                node_id: list(trucks_data.keys())
                for (node_id, trucks_data) in node_trucks_data.items()
                if trucks_data is not None
            }

            nx.draw_networkx_labels(
                self.graph,
                pos=truck_label_pos,
                labels=node_trucks_labels,
                ax=ax,
                font_size=8,
            )

            # trailers
            label_offset = np.array([0, 0.2])
            trailer_label_pos = {k: (v - 0.2 * label_offset) for k, v in pos.items()}

            node_trailers_data = nx.get_node_attributes(self.graph, "trailers")
            node_trailers_labels = self.get_trailer_labels(node_trailers_data)

            nx.draw_networkx_labels(
                self.graph,
                pos=trailer_label_pos,
                labels=node_trailers_labels,
                ax=ax,
                font_size=8,
            )

            # draw edges
            for edge in self.graph.edges(data=True, keys=True):
                _, _, key, data = edge
                if "truck" in data and "trailer" in data and "timestamp" in data:
                    truck, trailer, timestamp = (
                        data["truck"],
                        data["trailer"],
                        data["timestamp"],
                    )

                    data["color"] = color[trailer]
                    data["style"] = style[truck]
                    data["label"] = str(timestamp)

            for u, v, key, data in self.graph.edges(data=True, keys=True):
                if "truck" in data and "trailer" in data and "timestamp" in data:
                    pos_source = pos[u]
                    pos_target = pos[v]

                    if key % 2 == 0:
                        offset = 0.1
                    else:
                        offset = -0.1

                    x_control, y_control = self.bezier_control_point(
                        pos_source, pos_target, offset
                    )

                    arrow = FancyArrowPatch(
                        pos_source,
                        pos_target,
                        connectionstyle=f"arc3, rad={offset}",
                        arrowstyle="->, head_length=0.5, head_width=0.3",
                        linestyle=data["style"],
                        linewidth=1,
                        color=data["color"],
                        zorder=-key,
                        alpha=0.7,
                        mutation_scale=20,
                    )

                    ax.add_patch(arrow)

                    x_label = (pos_source[0] + x_control) / 2
                    y_label = (pos_source[1] + y_control) / 2
                    plt.text(
                        x_label,
                        y_label,
                        data["label"],
                        fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.5),
                    )

            plt.show()

    def visit_edge(self, data: list) -> None:
        """
        Add the visited edges.

        Args:
            source_node (int): Source node id of the edge
            target_node (int): Target node id of the edge
            truck (src): Truck id of the edge
            trailer (src): Trailer id of the edge
            timestamp (src): Timestamp id of the edge
        """
        for source_node, target_node, truck, trailer, timestamp in data:
            trucks = self.graph.nodes.data()[source_node]["trucks"]
            trailers = self.graph.nodes.data()[source_node]["trailers"]

            if not bool(trucks) or (bool(trucks) and truck not in trucks.keys()):
                # do not add an edge when the truck is not already in the source node
                continue
            elif not bool(trailers) or (
                bool(trailers) and trailer not in trailers.keys()
            ):
                # do not add trailer when it is not already in the source node, move just the truck
                trailer = None

            self.graph.add_edges_from(
                [
                    (
                        source_node,
                        target_node,
                        {"truck": truck, "trailer": trailer, "timestamp": timestamp},
                    )
                ]
            )

    @property
    def _num_chargers(self) -> np.ndarray:
        positions = nx.get_node_attributes(self.graph, "num_chargers").values()
        return np.asarray(list(positions))

    @property
    def _edges(self):
        return self.graph.edges.data()

    @property
    def _nodes(self):
        return self.graph.nodes.data()

    @property
    def _graph(self):
        return self.graph

    @property
    def _node_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node as
        an ndarray of shape (num_nodes, 2) sorted
        by the node index.
        """

        positions = nx.get_node_attributes(self.graph, "coordinates").values()
        return np.asarray(list(positions))

    def euclid_distance(self, node1_idx: int, node2_idx: int) -> float:
        """
        Calculates the euclid distance between two nodes
        with their idx's respectively.
        """

        node_one_pos = self.graph.nodes[node1_idx]["coordinates"]
        node_two_pos = self.graph.nodes[node2_idx]["coordinates"]

        return np.linalg.norm(node_one_pos - node_two_pos)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    G = EVRPGraph(num_nodes=4, num_trailers=3, num_trucks=2, plot_attributes=True)
    # add edges that where visited
    edges = [
        (0, 3, "Truck 1", "Trailer B", 1),
        (0, 3, "Truck 2", None, 2),
        (3, 2, "Truck 1", "Trailer A", 3),
        (3, 2, "Truck 2", "Trailer C", 4),
    ]

    G.visit_edge(edges)

    G.draw(ax=ax, with_labels=True)
    print(G._edges)
    print(G._nodes)
