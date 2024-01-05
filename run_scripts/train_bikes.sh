#!/bin/bash

group_name="new_bikes_test"

python -m run_scripts.train overrides=pets_bikes overrides.env_config.bikes_per_truck=5 experiment.run_configs.name=n_bikes_5 experiment.run_configs.group=${group_name} 
python -m run_scripts.train overrides=pets_bikes overrides.env_config.bikes_per_truck=10  experiment.run_configs.name=n_bikes_10 experiment.run_configs.group=${group_name} 
python -m run_scripts.train overrides=pets_bikes overrides.env_config.bikes_per_truck=15 experiment.run_configs.name=n_bikes_15 experiment.run_configs.group=${group_name} 
python -m run_scripts.train overrides=pets_bikes overrides.env_config.bikes_per_truck=20 experiment.run_configs.name=n_bikes_20 experiment.run_configs.group=${group_name} 
#python -m run_scripts.train overrides=pets_bikes overrides.env_config.bikes_per_truck=10 overrides.env_config.centroids_coord=null experiment.run_configs.name=random_centroid experiment.run_configs.group=${group_name} 
