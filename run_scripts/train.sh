#Training bash script for sequential multiple runs
python -m run_scripts.train dynamics_model=factored_gaussian_mlp_ensemble experiment.run_configs.name=factored_5dim
python -m run_scripts.train dynamics_model=gaussian_mlp_ensemble experiment.run_configs.name=non_factored_5dim
python -m run_scripts.train dynamics_model=lasso_gaussian_mlp_ensemble experiment.run_configs.name=lasso_5dim