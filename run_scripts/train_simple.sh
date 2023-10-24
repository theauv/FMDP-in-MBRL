#!/bin/bash

group_name="simple_2d_small"

python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=lasso_simple experiment.run_configs.name=lasso
python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=simple experiment.run_configs.name=non_factored
python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=factored_simple experiment.run_configs.name=factored
