#!/bin/bash

group_name="simple_2d_small"

python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=gaussian_mlp_ensemble dynamics_model.model.deterministic=True experiment.run_configs.name=non_factored_gaussian
python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=simple experiment.run_configs.name=non_factored
python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=factored_simple experiment.run_configs.name=factored
