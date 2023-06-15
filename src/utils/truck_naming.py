import string
import numpy as np
import pandas as pd
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd)
FILE = "../problems/evrp/truck_names.csv"


def get_truck_names(file: str = None) -> list:
    # change this function with the custom name. By default, we only 52 names for trucks.

    if isinstance(file, list):
        return file

    if file is None:
        names = list(string.ascii_letters.swapcase())
    else:
        # Read the CSV file
        df = pd.read_csv(file, header=None)
        # Convert the DataFrame to a numpy array
        names = list(np.squeeze(df.values))

    return names


def get_truck_number(truck: str = "", file: str = None):
    names = get_truck_names(file)

    # Extract the letter from the string (assuming the format is always "Truck X")
    letter = truck.split(" ")[1]
    assert letter in names, "Wrong truck name"
    # Get the number of the truck
    number = names.index(letter)

    return number


def get_trailer_number(trailer: str = ""):
    # Extract the number from the string (assuming the format is always "Trailer X")
    number = trailer.split(" ")[1]

    return int(number)


if __name__ == "__main__":
    print(get_truck_number("Trailer Robin", FILE))
