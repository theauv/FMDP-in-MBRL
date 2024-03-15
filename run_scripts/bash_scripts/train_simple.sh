#!/bin/bash

group_name="ffnn_5d"

python -m run_scripts.train dynamics_model=lassonet dynamics_model.model_trainer.reinit=True experiment.run_configs.name=lasso experiment.run_configs.group=${group_name} 
python -m run_scripts.train dynamics_model=lassonet dynamics_model.model_trainer.reinit=False  experiment.run_configs.name=lasso experiment.run_configs.group=${group_name} 
python -m run_scripts.train dynamics_model=ffnn experiment.run_configs.name=non_factored experiment.run_configs.group=${group_name} 
python -m run_scripts.train dynamics_model=factored_ffnn experiment.run_configs.name=factored experiment.run_configs.group=${group_name} 
