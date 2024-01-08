"""
This code has been copied from acbo code on github: 
https://github.com/ssethz/acbo/blob/main/scripts/bikes_map_plotter.py

Plots on a map details of where bikes are being placed etc. 

The main function generates a plot of map containing the centroids coloured by cluster in the
BikeSparse environment, and a file containing cluster centers and the cluster label of each centroid.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import pickle
import matplotlib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import torch
import geopy.distance

from src.util.plot_utils import set_size, set_up_plt

set_up_plt(font_family="Times New Roman")
set_size(width="thesis")


def map_plot(
    lats, longs, s, title=None, save_path=None, centroids=None, met=None, unmet=None
):
    """
    Take latitudes, longitudes, and s (size of the point) and plot them as points on a map with strength alpha. 
    If centroids=None is overridden by the list of centroids then they are plotted as well. Same for coordinates of met and unmet trips
    """
    fig, ax = plt.subplots()

    if unmet is not None:
        try:
            unmet = torch.tensor(unmet, requires_grad=False)
            ax.scatter(
                unmet[:, 1],
                unmet[:, 0],
                s=1,
                zorder=1,
                color="red",
                alpha=0.3,
                rasterized=True,
            )
        except:
            None

    if met is not None:
        try:
            met = torch.tensor(met, requires_grad=False)
            ax.scatter(
                met[:, 1],
                met[:, 0],
                s=1,
                zorder=1,
                color="green",
                alpha=0.1,
                rasterized=True,
            )
        except:
            None

    if centroids is not None:
        centroids = torch.tensor(centroids, requires_grad=False)
        ax.scatter(
            centroids[:, 1],
            centroids[:, 0],
            s=10,
            zorder=2,
            color="black",
            alpha=1.0,
            rasterized=True,
            marker="x",
        )

    ax.scatter(longs, lats, s=s, color="blue", zorder=3, alpha=0.5, rasterized=True)

    city_map = plt.imread("src/env/bikes_data/louisville_map.png")
    ax.imshow(city_map, zorder=0, extent=[-85.9, -85.55, 38.15, 38.35], aspect="equal")
    ax.grid(False)
    plt.xlim(longs)
    plt.ylim(lats)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    if title is not None:
        plt.title(title)

    fig.tight_layout()
    if save_path is not None:
        filename = "scripts/map_plots" + save_path
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches="tight", dpi=450)
    return fig


def plot_bikes_timestep(bikes, centroids, t=0, algo_str="", met=None, unmet=None):
    """
    For a single timestep takes the bikes and centroids, then plots them on the graph. 
    """
    lats, longs = bikes_to_centroids(bikes, centroids)
    map_plot(
        lats,
        longs,
        s=20,
        title=f"Bikes at timestep {t} for {algo_str}",
        save_path=f"/{algo_str}/bikes_timestep_{t}.png",
        centroids=centroids,
        met=met,
        unmet=unmet,
    )


def bikes_to_centroids(bikes, centroids):
    """
    Takes an array of bike indexes and returns lats and longs by indexing centroids, which is a list of centroid locations.  
    """
    centroids = torch.tensor(centroids, requires_grad=False)

    lats = centroids[bikes.long()][:, 0]
    longs = centroids[bikes.long()][:, 1]

    return lats, longs


def weights_to_plot_data(weights, centroids):
    """
    Take weights and centroids and return the lats longs s for plotting on map. 
    """
    centroids = torch.tensor(centroids, requires_grad=False)
    lats = centroids[:, 0]
    longs = centroids[:, 1]
    s = weights.detach().numpy() * 50
    return lats, longs, s


def plot_weights_timestamp(weights, centroids, t=0, algo_str="", truck_num=0):
    """
    For a single timestep takes the weights and centroids, then plots the weight of each centroid on the map.  
    """
    lats, longs, s = weights_to_plot_data(weights, centroids)
    map_plot(
        lats,
        longs,
        s=s,
        title=f"Weights at timestep {t} and truck {truck_num} for {algo_str}",
        save_path=f"/{algo_str}/weights_timestep_{t}_{truck_num}.png",
        centroids=centroids,
    )


def plot_bike_locations(X, centroids, algo_str, t=0, met=None, unmet=None):
    """
    Takes a list of bike locations for each time chunk then plots on the graph where bikes are at each timechunk.
    """
    X = torch.tensor(X, requires_grad=False)
    X = X / X.sum(dim=1, keepdim=True)
    for i in range(len(X)):
        lats, longs, s = weights_to_plot_data(X[i], centroids)
        map_plot(
            lats,
            longs,
            s=10 * s,
            title=f"Bikes at timestep {i} for {algo_str} on day {t}",
            save_path=f"/{algo_str}/bikes_timestep_{t}_{i}_.png",
            centroids=centroids,
            met=met,
            unmet=unmet,
        )


def full_trial_map_plotter(X, centroids, algo_str, t=0, met=None, unmet=None):
    """
    The plots for the paper that show average bike locations over the year with all demand data for the year. 
    """
    trips_data = pd.read_csv(
        "scripts/bikes_data/dockless-vehicles-3_full.csv",
        usecols=lambda x: x not in ["TripID", "StartDate", "EndDate", "EndTime"],
    )
    trips_data = trips_data[
        ["StartLatitude", "StartLongitude", "EndLatitude", "EndLongitude"]
    ]
    met = trips_data[["StartLatitude", "StartLongitude"]].values

    X = torch.tensor(X, requires_grad=False)
    X = X / X.sum(dim=1, keepdim=True)
    for i in range(len(X)):
        lats, longs, s = weights_to_plot_data(X[i], centroids)
        map_plot(
            lats,
            longs,
            s=10 * s,
            title=None,
            save_path=f"/{algo_str}.pdf",
            centroids=centroids,
            met=met,
            unmet=unmet,
        )


def centroid_cluster(centroids, clusters):
    """
    Performs clustering on the centroids, then plots centroids as different locations depending on their group number. 
    """
    centroids = torch.tensor(centroids, requires_grad=False)

    from sklearn.cluster import KMeans

    fig, ax = plt.subplots()
    loose_centroids = []
    for i in range(len(centroids)):
        if i not in [item for sublist in clusters for item in sublist]:
            loose_centroids.append(i)

    for i in range(len(clusters)):
        if len(clusters[i]) == 1:
            loose_centroids.append(clusters[i][0])
            clusters[i] = []

    clusters = [x for x in clusters if x != []]
    for i in range(len(loose_centroids)):
        closest = 0
        closest_dist = 1000000
        for j in range(len(clusters)):
            dist = torch.norm(
                centroids[clusters[j]] - centroids[loose_centroids[i]], dim=1
            ).min()

            if dist < closest_dist:
                closest = j
                closest_dist = dist
        clusters[closest].append(loose_centroids[i])

    n_clusters = len(clusters)

    labels = []
    for i in range(len(centroids)):
        for j in range(len(clusters)):
            if i in clusters[j]:
                labels.append(j)
                break
    labels = np.array(labels)

    centers = []
    for i in range(len(clusters)):
        centers.append(centroids[clusters[i]].mean(dim=0))
    centers = torch.stack(centers)

    save_path = "/centroids_clustered_paper.png"

    for i in range(centroids.shape[0]):
        ax.scatter(
            centroids[i, 1],
            centroids[i, 0],
            s=25,
            zorder=2,
            color=cm.jet(labels[i] / len(clusters)),
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    city_map = plt.imread("scripts/map/louisville_map.png")

    ax.imshow(city_map, zorder=0, extent=[-85.9, -85.55, 38.15, 38.35], aspect="equal")
    ax.grid(False)
    plt.xlim([-85.9, -85.55])
    plt.ylim([38.15, 38.35])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    fig.tight_layout()
    fig.savefig("scripts/map_plots" + save_path, bbox_inches="tight")

    labels = np.array(labels)
    centers = np.array(centers)
    pickle.dump(
        (labels, centers),
        open(
            "scripts/bikes_data/clustered_centroids_"
            + str(len(centroids))
            + "_"
            + str(n_clusters)
            + ".pckl",
            "wb",
        ),
    )


if __name__ == "__main__":
    _, _, _, _, _, _, centroid_coords = pickle.load(
        open("scripts/bikes_data/training_data_" + "5" + "_" + "40" + ".pckl", "rb")
    )
    clusters = [
        [29],
        [34],
        [41],
        [76],
        [81],
        [89],
        [90],
        [115],
        [64, 59],
        [108, 97, 93, 100, 114, 39, 36, 104, 98],
        [94, 85],
        [74, 56, 83, 80, 3, 40, 99, 6, 84],
        [21, 38, 68, 17, 111, 9, 7, 25, 70],
        [69, 112, 48, 105, 52, 73, 53, 103, 87],
        [35, 82, 49, 13, 66, 30, 15, 2, 46],
        [86, 12, 26, 8, 24, 63, 62, 5, 20],
        [79, 110, 61, 22, 101, 96, 31],
        [51, 4],
        [102, 91, 92, 106, 60, 54, 109, 10],
        [43, 72, 107, 47, 1],
        [32, 27, 55, 19, 18, 42, 14, 16],
        [75, 37, 28, 11, 77, 44, 23, 45, 50],
        [65, 33, 88, 58, 113, 57, 78, 0],
    ]
    centroid_cluster(centroid_coords, clusters)
