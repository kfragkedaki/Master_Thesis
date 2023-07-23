import json
import numpy as np


def load_json(input_file=""):
    # Load the data from the JSON file
    with open(input_file, "r") as f:
        data = json.load(f)
    if validate_data(data):
        return trasform_data(data)
    else:
        return None


def validate_data(data):
    # Check nodes
    node_ids = [node["id"] for node in data["nodes"]]
    if not all(isinstance(node_id, int) for node_id in node_ids):
        return False
    if max(node_ids) != len(node_ids) - 1:
        return False

    for node in data["nodes"]:
        if not (0 <= node["x"] <= 1) or not (0 <= node["y"] <= 1):
            return False
        if node["num_chargers"] <= 0:
            return False

    # Check trailers
    trailer_ids = [int(trailer["id"].split(" ")[1]) for trailer in data["trailers"]]
    if sorted(trailer_ids) != list(range(len(trailer_ids))):
        return False

    for i, trailer in enumerate(data["trailers"]):
        if trailer["id"] != f"Trailer {i}":
            return False
        if (
            trailer["originNodeId"] not in node_ids
            or trailer["destinationNodeId"] not in node_ids
        ):
            return False

    # Check trucks
    for truck in data["trucks"]:
        if not truck["id"].startswith("Truck "):
            return False
        if truck["startBatteryLevel"] != 1:
            return False
        if truck["originNodeId"] not in node_ids:
            return False

    return True


def trasform_data(data: dict):
    # Initialize an empty dictionary for the transformed data
    transformed_data = {}

    # Process the nodes
    for node in data["nodes"]:
        transformed_data[node["id"]] = {
            "coordinates": np.array([node["x"], node["y"]]),
            "num_chargers": node["num_chargers"],
            "available_chargers": node[
                "num_chargers"
            ],  # all chargers are available initially
            "trailers": {},
            "trucks": {},
        }

    # Process the trailers
    for trailer in data["trailers"]:
        origin_node = trailer["originNodeId"]
        trailer_info = {
            "destination_node": trailer["destinationNodeId"],
            "start_time": trailer["startTime"],
            "end_time": trailer["endTime"],
        }
        transformed_data[origin_node]["trailers"][trailer["id"]] = trailer_info

    # Process the trucks
    for truck in data["trucks"]:
        origin_node = truck["originNodeId"]
        truck_info = {"battery_level": truck["startBatteryLevel"]}
        transformed_data[origin_node]["trucks"][truck["id"]] = truck_info

    # Set trucks and trailers to None if empty
    for node_id, node_data in transformed_data.items():
        if not node_data["trailers"]:
            node_data["trailers"] = None
        if not node_data["trucks"]:
            node_data["trucks"] = None

    return transformed_data


def get_information_from_dict(input: dict) -> dict:
    data = {}
    data["num_nodes"] = len(input)
    data["num_trucks"] = sum(
        len(graph.get("trucks", {})) if graph.get("trucks") is not None else 0
        for graph in input.values()
    )
    data["num_trailers"] = sum(
        len(graph.get("trailers", {})) if graph.get("trailers") is not None else 0
        for graph in input.values()
    )
    data["truck_names"] = [
        truck_name.split(" ")[1]
        for graph in input.values()
        if graph.get("trucks") is not None
        for truck_name in graph["trucks"]
    ]

    return data


if __name__ == "__main__":
    result = load_json("src/instances/example.json")
    print(result)
