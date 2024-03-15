#!/bin/bash

module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .



centroids="src/env/bikes_data/5_centroids/5_centroids.npy"
for sparsity in null "src/env/bikes_data/5_centroids/factors_radius_4.npy"
    do
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/create_datasets.py dataset_folder_name='art_datasets' overrides=pets_bikes debug_mode=false overrides.env_config.centroids_coord=${centroids}  overrides.env_config.station_dependencies=${sparsity}"
done
