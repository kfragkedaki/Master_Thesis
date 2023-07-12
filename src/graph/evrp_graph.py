import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from src.utils.load_json import get_information_from_dict
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
        if data is not None:
            nx.set_node_attributes(self.graph, "lightblue", "node_color")
            nx.set_node_attributes(self.graph, data)
        elif len(kwargs) > 0:
            self.set_default_attributes(**kwargs)
        else:
            self.set_default_attributes()

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
            coords = self._compute_coordinates()
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
            }

        # set trucks
        trucks = get_truck_names(file=self.truck_names)

        for i, node_id in enumerate(trucks_locations):
            if self.graph.nodes[node_id]["trucks"] is None:
                self.graph.nodes[node_id]["trucks"] = {}
            self.graph.nodes[node_id]["trucks"][f"Truck {trucks[i]}"] = {
                "battery_level": trucks_battery_levels
            }

    def _compute_coordinates(self):
        def check_distance(new_point, points, r_threshold):
            for point in points.values():
                dist = 0
                if point is not None:
                    dist = np.sqrt(np.sum(np.square(new_point - point)))
                if dist > r_threshold:
                    return False
            return True

        coordinates = dict.fromkeys(range(self.num_nodes))
        idx = 0
        while any(value is None for value in coordinates.values()):
            new_point = np.random.rand(2)
            if check_distance(new_point, coordinates, r_threshold=0.6):
                coordinates[idx] = new_point
                idx += 1

        return coordinates

    def get_trailer_labels(self, data, colors):
        node_trailers = {}
        node_colors = {}
        for node_id, trailers_data in data.items():
            if trailers_data is not None:
                # origin
                if node_id not in node_trailers:
                    node_trailers[node_id] = ""

                node_trailers[node_id] += str(list(trailers_data.keys())) + "\n"

                for trailer_id, trailer_data in trailers_data.items():
                    destination_node = trailer_data["destination_node"]

                    # destination
                    if destination_node not in node_colors:
                        node_colors[destination_node] = []

                    node_colors[destination_node].append(colors[trailer_id])

        return node_trailers, node_colors

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

    def draw_graph_with_multicolor_circles(self, pos, ax, node_colors_dict):
        # Draw concentric circles around the nodes
        for node, colors in node_colors_dict.items():
            x, y = pos[node]
            for i, color in enumerate(colors):
                circle = plt.Circle(
                    (x, y),
                    radius=(0.15 * (i / 10 + 0.5)),
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_artist(circle)

    def normalize_positions(self, pos):
        x_values, y_values = zip(*pos.values())

        x_min = min(x_values)
        x_max = max(x_values)
        y_min = min(y_values)
        y_max = max(y_values)

        pos_normalized = {}
        for node, (x, y) in pos.items():
            pos_normalized[node] = (
                (x - x_min) / (x_max - x_min),
                (y - y_min) / (y_max - y_min),
            )

        return pos_normalized

    def draw(self, ax, with_labels=True):
        """
        Draws the graph as a matplotlib plot.
        """
        graph_copy = copy.deepcopy(self.graph)

        # draw nodes according to color and position attribute
        pos = nx.get_node_attributes(graph_copy, "coordinates")
        pos = self.normalize_positions(pos)

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
            label_offset = np.array([0, 0.37])
            truck_label_pos = {k: (v - 0.4 * label_offset) for k, v in pos.items()}

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
            node_trailers_labels, node_colors_dict = self.get_trailer_labels(
                node_trailers_data, color
            )

            nx.draw_networkx_labels(
                graph_copy,
                pos=trailer_label_pos,
                labels=node_trailers_labels,
                ax=ax,
                font_size=8,
            )

            self.draw_graph_with_multicolor_circles(pos, ax, node_colors_dict)

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

            if source_node == -1 or target_node == -1 or truck == -1 or truck == None:
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
            truck_names = get_truck_names(file=self.truck_names)
            truck_id = f"Truck {truck_names[int(truck)]}"
            assert not (
                not bool(trucks) or (bool(trucks) and truck_id not in trucks.keys())
            ), f"The truck is not in the source node, {truck_id}, {trucks}"

            self.get_neighbors(source_node)
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

    def clear(self):
        edges = list(self.graph.edges())
        self.graph.remove_edges_from(edges)
        nx.set_node_attributes(self.graph, "lightblue", "node_color")

    def euclid_distance(self, node1_idx: int, node2_idx: int) -> float:
        """
        Calculates the euclid distance between two nodes
        with their idx's respectively.
        """

        node_one_pos = self.graph.nodes[node1_idx]["coordinates"]
        node_two_pos = self.graph.nodes[node2_idx]["coordinates"]

        return np.linalg.norm(node_one_pos - node_two_pos)

    def get_neighbors(self, cur_node, r_threshold=0.6) -> np.ndarray:
        nns = []
        for node in self.graph.nodes():
            if node != cur_node and self.euclid_distance(cur_node, node) <= r_threshold:
                self.graph.nodes[node]["node_color"] = "lightgray"
                nns.append(node)

        return nns


EXAMPLE_GRAPH = {
    0: {
        "coordinates": np.array([0.81568734, 0.46616891]),
        "num_chargers": 2,
        "available_chargers": 2,
        "trailers": {
            "Trailer 0": {"destination_node": 3, "start_time": 14, "end_time": 15.0},
            "Trailer 2": {"destination_node": 3, "start_time": 9, "end_time": 10.0},
        },
        "trucks": None,
    },
    1: {
        "coordinates": np.array([0.45070739, 0.77785083]),
        "num_chargers": 9,
        "available_chargers": 9,
        "trailers": None,
        "trucks": {"Truck A": {"battery_level": 1}},
    },
    2: {
        "coordinates": np.array([0.06070113, 0.53816134]),
        "num_chargers": 7,
        "available_chargers": 7,
        "trailers": None,
        "trucks": {"Truck B": {"battery_level": 1}},
    },
    3: {
        "coordinates": np.array([0.39932656, 0.17680608]),
        "num_chargers": 8,
        "available_chargers": 8,
        "trailers": {
            "Trailer 1": {"destination_node": 2, "start_time": 16, "end_time": 17.5}
        },
        "trucks": None,
    },
}

if __name__ == "__main__":
    result = get_information_from_dict(EXAMPLE_GRAPH)
    G = EVRPGraph(
        num_nodes=result["num_nodes"],
        num_trailers=result["num_trailers"],
        num_trucks=result["num_trucks"],
        truck_names=result["truck_names"],
        plot_attributes=True,
        data=EXAMPLE_GRAPH,
    )
    # add edges that where visited
    edges = [(1, 0, 0, None, 1), (2, 0, 1, None, 2), (0, 3, 0, 2, 3)]

    print(G._nodes)

    for edge in edges:
        fig, ax = plt.subplots()

        G.draw(ax=ax, with_labels=True)
        G.clear()
        edge = G.visit_edge([edge])
        G.update_attributes(edge)
        plt.axis("equal")
        plt.show()

    fig, ax = plt.subplots()
    G.draw(ax=ax, with_labels=True)
    plt.axis("equal")
    plt.show()

    print(G._edges)
    print(G._nodes)
