"""
This file contains functions that create an adjacency matrix based on certain 
criteria. Therefore obtaining factors that can be induced in the Bikes environment 
"""

import argparse
import os

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
from scipy.spatial.distance import cdist
import geopy.distance


def create_local_factor(centroid_coords, radius):
    num_centroids = len(centroid_coords)
    adjacency = np.zeros((num_centroids, num_centroids))

    for centroid_i, coords_i in enumerate(centroid_coords):
        for centroid_j, coords_j in enumerate(centroid_coords):
            distance = geopy.distance.distance(coords_i, coords_j).km
            if distance <= radius:
                adjacency[centroid_i, centroid_j] = 1

    return adjacency


def draw_map_factors(centroid_coords, adjacency):
    # TODO: draw transparent radius around nodes to show the actual radius?

    graph_size = 600
    net = Network(f"{graph_size}px", select_menu=True)
    net.toggle_physics(False)

    for i, centroid_coord in enumerate(centroid_coords):
        centroid_coord = np.flip(centroid_coord)
        net.add_node(
            i,
            label=i,
            x=centroid_coord[0] * graph_size**2,
            y=(graph_size - centroid_coord[1]) * graph_size**2,
            color="blue",
            size=200,
        )
    for centroid_i, factor_i in enumerate(adjacency):
        for centroid_j, scope_j in enumerate(factor_i):
            if scope_j > 0.0 and centroid_i != centroid_j:
                net.add_edge(centroid_i, centroid_j, width=0.1)

    for n in net.nodes:
        n["font"] = {"size": 1000}

    net.show("adjacency_factor_vis.html", notebook=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--centroid_coords_file",
        type=str,
        default="src/env/bikes_data/LouVelo_centroids_coords.npy",
        help="File containing the coordinates of the centroids",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3,
        help="Maximum trip distance starting from any centroid (in km)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use it if you just want to test the sparsity without storing the adjacency",
    )
    args = parser.parse_args()

    centroid_coords = np.load(args.centroid_coords_file)
    adjacency = create_local_factor(centroid_coords, args.radius)
    filename = f"factors_radius_{int(args.radius)}"
    directory = os.path.dirname(os.path.relpath(__file__))
    file_path = os.path.join(directory, filename)
    print("Adjacency", adjacency)
    ratio = round(np.count_nonzero(adjacency) / (len(adjacency) ** 2), 3) * 100
    print(f"Sparsity: {ratio}% of non zeros")
    draw_map_factors(centroid_coords, adjacency)
    if not args.test:
        np.save(file_path, adjacency)
