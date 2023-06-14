import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from src.utils.truck_naming import get_truck_names
from matplotlib.lines import Line2D
import copy


class EVRPGraph:
    graph: nx.Graph = nx.Graph()

    def __init__(
        self,
        num_nodes: int,
        num_trailers: int,
        num_trucks: int,
        truck_names: str = None,
        plot_attributes: bool = True,
        data: dict() = None,
        **kwargs,
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
        self.truck_names = truck_names
        self.plot_attributes = plot_attributes

        # generate graph and set node position
        self.graph = nx.complete_graph(num_nodes, nx.MultiDiGraph())
        if data is None:
            self.set_default_attributes(**kwargs)
        else:
            nx.set_node_attributes(self.graph, data)
            nx.set_node_attributes(self.graph, "lightblue", "node_color")

    def set_default_attributes(
        self,
        coords=None,
        num_chargers=None,
        trailers_locations=None,
        trailers_start_time=None,
        trailers_end_time=None,
        trailers_destinations=None,
        trucks_locations=None,
        trucks_battery_levels=1,
        trailers_status=1,
    ):
        """
        Sets the default colors of the nodes
        as attributes. Nodes are black except
        depots which are colored in red.

        Edges are initially marked as unvisited.
        """

        # Node Attributes
        # If data is not provided, generate it
        if coords is None:
            coords = dict(enumerate(np.random.rand(self.num_nodes, 2)))
        if num_chargers is None:
            num_chargers = dict(
                enumerate(np.random.randint(low=1, high=5, size=self.num_nodes))
            )
        if trailers_locations is None:
            trailers_locations = np.random.choice(self.graph.nodes, self.num_trailers)
        if trailers_start_time is None:
            trailers_start_time = np.random.randint(
                low=8.00, high=18.00, size=self.num_trailers
            )
        if trailers_end_time is None:
            trailers_end_time = (
                trailers_start_time
                + np.round(
                    np.random.uniform(low=0.5, high=2, size=self.num_trailers) * 2
                )
                / 2
            )
        if trailers_destinations is None:
            trailers_destinations = np.array(
                [
                    np.random.choice([n for n in self.graph.nodes if n != node_id])
                    for node_id in trailers_locations
                ]
            )
        if trucks_locations is None:
            trucks_locations = np.random.choice(self.graph.nodes, self.num_trucks)

        # set attributes for each node
        nx.set_node_attributes(self.graph, coords, name="coordinates")
        nx.set_node_attributes(self.graph, num_chargers, name="num_chargers")
        nx.set_node_attributes(self.graph, num_chargers, name="available_chargers")
        nx.set_node_attributes(self.graph, None, name="trailers")
        nx.set_node_attributes(self.graph, None, name="trucks")
        nx.set_node_attributes(self.graph, "lightblue", "node_color")

        for i, node_id in enumerate(trailers_locations):
            if self.graph.nodes[node_id]["trailers"] is None:
                self.graph.nodes[node_id]["trailers"] = {}
            self.graph.nodes[node_id]["trailers"][f"Trailer {i}"] = {
                "destination_node": trailers_destinations[i],
                "start_time": np.round(trailers_start_time[i], 2),
                "end_time": trailers_end_time[i],
                "status": trailers_status,  # 1: "Available", 0: "Pending"
            }

        # set trucks
        trucks = get_truck_names(file=self.truck_names)

        for i, node_id in enumerate(trucks_locations):
            if self.graph.nodes[node_id]["trucks"] is None:
                self.graph.nodes[node_id]["trucks"] = {}
            self.graph.nodes[node_id]["trucks"][f"Truck {trucks[i]}"] = {
                "battery_level": trucks_battery_levels
            }

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

    def add_legend(self, trailers, trucks):
        plt.subplots_adjust(right=0.8)
        keys = []
        custom_lines = []

        for key, value in trailers.items():
            if key is None:
                key = "No Trailer"
            keys.append(key)
            custom_lines.append(Line2D([0], [0], color=value, lw=2))

        for key, value in trucks.items():
            if key is None:
                continue
            keys.append(key)
            custom_lines.append(Line2D([0], [0], color="black", lw=2, ls=value))

        plt.legend(custom_lines, keys, loc="upper right", bbox_to_anchor=(1.3, 1))

    def draw(self, ax, with_labels=True):
        """
        Draws the graph as a matplotlib plot.
        """
        graph_copy = copy.deepcopy(self.graph)

        # draw nodes according to color and position attribute
        pos = nx.get_node_attributes(graph_copy, "coordinates")
        node_colors = nx.get_node_attributes(graph_copy, "node_color").values()
        nx.draw_networkx_nodes(
            graph_copy,
            pos=pos,
            node_color=node_colors,
            ax=ax,
            node_size=400,
            margins=0.5,
        )

        # draw attributes
        if self.plot_attributes:
            colors = plt.cm.rainbow(np.linspace(0, 1, self.num_trailers))
            color = {f"Trailer {i}": colors[i] for i in range(self.num_trailers)}
            color[None] = "black"

            truck_names = get_truck_names(self.truck_names)
            styles = ["solid"] + [
                (0, (i + 1, self.num_trucks - i)) for i in range(self.num_trucks)
            ]
            style = {
                f"Truck {truck_names[i]}": styles[i % len(styles)]
                for i in range(self.num_trucks)
            }
            style[
                None
            ] = "solid"  # this should never appear, since we cannot move without a truck

            if with_labels:
                self.add_legend(color, style)

            # chargers
            node_num_chargers = nx.get_node_attributes(graph_copy, "num_chargers")
            node_available_chargers = nx.get_node_attributes(
                graph_copy, "available_chargers"
            )

            node_num_chargers = {
                node_id: f"{avail_charg} / {num_charg}, \n {node_id}"
                for (node_id, num_charg), avail_charg in zip(
                    node_num_chargers.items(), node_available_chargers.values()
                )
            }
            nx.draw_networkx_labels(
                graph_copy, pos=pos, labels=node_num_chargers, ax=ax, font_size=6
            )
            # trucks
            label_offset = np.array([0, 0.09])
            truck_label_pos = {k: (v - 0.6 * label_offset) for k, v in pos.items()}

            node_trucks_data = nx.get_node_attributes(graph_copy, "trucks")
            node_trucks_labels = {
                node_id: list(trucks_data.keys())
                for (node_id, trucks_data) in node_trucks_data.items()
                if trucks_data is not None
            }

            nx.draw_networkx_labels(
                graph_copy,
                pos=truck_label_pos,
                labels=node_trucks_labels,
                ax=ax,
                font_size=8,
            )

            # trailers
            label_offset = np.array([0, 0.6])
            trailer_label_pos = {k: (v - 0.2 * label_offset) for k, v in pos.items()}

            node_trailers_data = nx.get_node_attributes(graph_copy, "trailers")
            node_trailers_labels = self.get_trailer_labels(node_trailers_data)

            nx.draw_networkx_labels(
                graph_copy,
                pos=trailer_label_pos,
                labels=node_trailers_labels,
                ax=ax,
                font_size=8,
            )

            # draw edges
            for edge in graph_copy.edges(data=True, keys=True):
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

            for u, v, key, data in graph_copy.edges(data=True, keys=True):
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

    def visit_edge(self, data: list = []) -> tuple:
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
            source_node = int(source_node)
            target_node = int(target_node)

            if source_node == -1 or target_node == -1:
                continue

            trucks = self.graph.nodes.data()[source_node]["trucks"]
            trailers = self.graph.nodes.data()[source_node]["trailers"]

            # check trailer
            if trailer == -1 or trailer == None:
                trailer_id = None
            else:
                trailer_id = f"Trailer {int(trailer)}"
                assert not (
                    not bool(trailers)
                    or (bool(trailers) and trailer_id not in trailers.keys())
                ), f"Trailer when it is not in the source node, {trailers}, {trailer_id}, {trailer}"

                assert not (
                    bool(trailers)
                    and source_node == trailers[trailer_id]["destination_node"]
                ), f"Trailer in destination node, {source_node}, {trailers}, {trailer_id}"

            # check truck
            if truck == -1 or truck == None:
                truck_id = None
            else:
                truck_names = get_truck_names(file=self.truck_names)
                truck_id = f"Truck {truck_names[int(truck)]}"

                assert not (
                    not bool(trucks) or (bool(trucks) and truck_id not in trucks.keys())
                ), f"The truck is not in the source node, {truck_id}, {trucks}"

            self.graph.add_edges_from(
                [
                    (
                        source_node,
                        target_node,
                        {
                            "truck": truck_id,
                            "trailer": trailer_id,
                            "timestamp": int(timestamp),
                        },
                    )
                ]
            )

            return (source_node, target_node, truck_id, trailer_id, int(timestamp))

    def update_attributes(self, edge: list) -> None:
        source_node, target_node, truck_id, trailer_id, timestep = edge
        chargers = self._node_chargers

        for node_index in range(self.num_nodes):  # reset
            avail_chargers = int(chargers[node_index])
            if node_index == target_node and truck_id is not None:
                avail_chargers = avail_chargers - 1

            self.graph.nodes[node_index]["available_chargers"] = avail_chargers

        if target_node != -1 and source_node != -1 and truck_id is not None:
            del self.graph.nodes[source_node]["trucks"][truck_id]
            if not bool(self.graph.nodes[source_node]["trucks"]):
                self.graph.nodes[source_node]["trucks"] = None
            data = {"battery_level": 0}
            if self.graph.nodes[target_node]["trucks"] is not None:
                self.graph.nodes[target_node]["trucks"][truck_id] = data
            else:
                self.graph.nodes[target_node]["trucks"] = {truck_id: data}

        if target_node != -1 and source_node != -1 and trailer_id is not None:
            data = self.graph.nodes[source_node]["trailers"][trailer_id]
            del self.graph.nodes[source_node]["trailers"][trailer_id]
            if not bool(self.graph.nodes[source_node]["trailers"]):
                self.graph.nodes[source_node]["trailers"] = None

            if self.graph.nodes[target_node]["trailers"] is not None:
                self.graph.nodes[target_node]["trailers"][trailer_id] = data
            else:
                self.graph.nodes[target_node]["trailers"] = {trailer_id: data}

    @property
    def _node_chargers(self) -> np.ndarray:
        chargers = nx.get_node_attributes(self.graph, "num_chargers").values()
        return np.asarray(list(chargers))

    @property
    def _node_avail_chargers(self) -> np.ndarray:
        available_chargers = nx.get_node_attributes(
            self.graph, "available_chargers"
        ).values()
        return np.asarray(list(available_chargers))

    @property
    def _node_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node as
        an ndarray of shape (num_nodes, 2) sorted
        by the node index.
        """

        positions = nx.get_node_attributes(self.graph, "coordinates").values()
        return np.asarray(list(positions))

    @property
    def _node_trailers(self) -> np.ndarray:
        """
        Returns trailers of each node as
        an ndarray of shape (num_nodes, 1) sorted
        by the node index.
        """

        trailers = nx.get_node_attributes(self.graph, "trailers").values()
        return np.asarray(list(trailers))

    @property
    def _node_trucks(self) -> np.ndarray:
        """
        Returns trucks of each node as
        an ndarray of shape (num_nodes, 2) sorted
        by the node index.
        """

        trucks = nx.get_node_attributes(self.graph, "trucks").values()
        return np.asarray(list(trucks))

    @property
    def _edges(self):
        return self.graph.edges.data()

    @property
    def _nodes(self):
        return self.graph.nodes.data()

    @property
    def _graph(self):
        return self.graph

    def remove_edges(self):
        edges = list(self.graph.edges())
        self.graph.remove_edges_from(edges)

    def euclid_distance(self, node1_idx: int, node2_idx: int) -> float:
        """
        Calculates the euclid distance between two nodes
        with their idx's respectively.
        """

        node_one_pos = self.graph.nodes[node1_idx]["coordinates"]
        node_two_pos = self.graph.nodes[node2_idx]["coordinates"]

        return np.linalg.norm(node_one_pos - node_two_pos)


EXAMPLE_GRAPH = {
    0: {
        "coordinates": [0.81568734, 0.46616891],
        "num_chargers": 2,
        "available_chargers": 2,
        "trailers": {
            "Trailer 0": {"destination_node": 3, "start_time": 14, "end_time": 15.0},
            "Trailer 2": {"destination_node": 3, "start_time": 9, "end_time": 10.0},
        },
        "trucks": None,
    },
    1: {
        "coordinates": [0.45070739, 0.77785083],
        "num_chargers": 9,
        "available_chargers": 9,
        "trailers": None,
        "trucks": {"Truck A": {"battery_level": 1}},
    },
    2: {
        "coordinates": [0.06070113, 0.53816134],
        "num_chargers": 7,
        "available_chargers": 7,
        "trailers": None,
        "trucks": {"Truck B": {"battery_level": 1}},
    },
    3: {
        "coordinates": [0.39932656, 0.17680608],
        "num_chargers": 8,
        "available_chargers": 8,
        "trailers": {
            "Trailer 1": {"destination_node": 2, "start_time": 16, "end_time": 17.5}
        },
        "trucks": None,
    },
}

if __name__ == "__main__":
    G = EVRPGraph(
        num_nodes=4,
        num_trailers=3,
        num_trucks=2,
        plot_attributes=True,
        data=EXAMPLE_GRAPH,
    )
    # add edges that where visited
    edges = [(1, 0, 0, None, 1), (2, 0, 1, None, 2), (0, 3, 0, 2, 3)]

    for edge in edges:
        fig, ax = plt.subplots()

        G.remove_edges()
        edge = G.visit_edge([edge])
        G.update_attributes(edge)
        G.draw(ax=ax, with_labels=True)
        plt.show(bbox_inches="tight")

    print(G._edges)
    print(G._nodes)
