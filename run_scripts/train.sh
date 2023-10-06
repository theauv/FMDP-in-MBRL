#Training bash script for sequential multiple runs
python -m run_scripts.train dynamics_model=factored_gaussian_mlp_ensemble
python -m run_scripts.train dynamics_model=gaussian_mlp_ensemble