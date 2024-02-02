#!/bin/bash

module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .


centroids="src/env/bikes_data/LouVelo_centroids/LouVelo_centroids_coords.npy"
for sparsity in null "src/env/bikes_data/LouVelo_centroids/factors_radius_3.npy" "src/env/bikes_data/LouVelo_centroids/factors_radius_1-2.npy"
    do
    for n_actions in 1 2 4
        do
        if (($n_actions == 1)) ; then
            bikes_per_truck=8
        elif (($n_actions == 2)) ; then
            bikes_per_truck=6
        else
            bikes_per_truck=3
        fi
        sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/create_datasets.py overrides=pets_bikes debug_mode=false overrides.env_config.centroids_coord=${centroids}  overrides.env_config.station_dependencies=${sparsity} overrides.env_config.action_per_day=${n_actions}"
    done
done

centroids="src/env/bikes_data/acbo_centroids/acbo_centroids_coords.npy"
for sparsity in null "src/env/bikes_data/acbo_centroids/factors_radius_5.npy" "src/env/bikes_data/acbo_centroids/factors_radius_3.npy"
    do
    for n_actions in 1 2 4
        do
        if (($n_actions == 1)) ; then
            bikes_per_truck=8
        elif (($n_actions == 2)) ; then
            bikes_per_truck=6
        else
            bikes_per_truck=3
        fi
        sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/create_datasets.py overrides=pets_bikes debug_mode=false overrides.env_config.centroids_coord=${centroids}  overrides.env_config.station_dependencies=${sparsity} overrides.env_config.action_per_day=${n_actions}"
    done
done