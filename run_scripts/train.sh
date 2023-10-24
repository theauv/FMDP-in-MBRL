#!/bin/bash
#Training bash script for sequential multiple runs

group_name="small_5dim_longertime"

python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=factored_gaussian_mlp_ensemble experiment.run_configs.name=factored_5dim
python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=gaussian_mlp_ensemble experiment.run_configs.name=non_factored_5dim
python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=lasso_gaussian_mlp_ensemble experiment.run_configs.name=lasso_5dim