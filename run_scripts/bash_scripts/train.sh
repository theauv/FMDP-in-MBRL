#!/bin/bash
#Training bash script for sequential multiple runs

initial_exploration_steps=10
group_name="factored_test_initial_${initial_exploration_steps}_steps"

python -m run_scripts.train dynamics_model=factored_ffnn experiment.run_configs.name=fixed_factored experiment.run_configs.group=${group_name} algorithm.initial_exploration_steps=${initial_exploration_steps}
python -m run_scripts.train dynamics_model=lassonet dynamics_model.model_trainer.reinit=True experiment.run_configs.name=fixed_lasso_reinit experiment.run_configs.group=${group_name} algorithm.initial_exploration_steps=${initial_exploration_steps}
python -m run_scripts.train dynamics_model=lassonet dynamics_model.model_trainer.reinit=False experiment.run_configs.name=lasso_no_reinit experiment.run_configs.group=${group_name} algorithm.initial_exploration_steps=${initial_exploration_steps}
python -m run_scripts.train dynamics_model=grid_factored_ffnn experiment.run_configs.name=grid_factored experiment.run_configs.group=${group_name} algorithm.initial_exploration_steps=${initial_exploration_steps}
#python -m run_scripts.train experiment.run_configs.group=${group_name} dynamics_model=lasso_gaussian_mlp_ensemble experiment.run_configs.name=lasso_5dim