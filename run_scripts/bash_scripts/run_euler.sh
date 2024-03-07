#!/bin/bash

# bash
# module load eth_proxy
# module load gcc/8.2.0 python/3.9.9
# conda activate pdm-env
# pip install -e .

group_name='art_4step'
overrides='pets_bikes'
rescale_input=true
rescale_output=true
target_is_delta=true
for i in {1..3}
    do
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
    for model in 'factored_gp' 'gaussian_process'
        do
        for obs_postprocess_fn in 'obs_postprocess_pred_proba' 'obs_postprocess_fn'
            do
            run_name="${model}_${obs_postprocess_fn}_targetdelta_${target_is_delta}_${i}"
            sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
        done
    done
done

group_name='real_4step'
overrides='pets_bikes'
trips_data="src/env/bikes_data/all_trips_LouVelo_merged.csv"
weather_data="src/env/bikes_data/weather_data.csv"
for i in {1..3}
    do
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
    for model in 'factored_gp' 'gaussian_process'
        do
        for obs_postprocess_fn in 'obs_postprocess_pred_proba' 'obs_postprocess_fn'
            do
            run_name="${model}_${obs_postprocess_fn}_targetdelta_${target_is_delta}_${i}"
            sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
        done
    done
done



#######################################################
# group_name='hypergrid'
# overrides='pets_hypergrid'
# lr=0.01
# num_epochs_train_model=30
# target_is_delta=true
# initial_exploration_steps=100
# for i in {1..3}
#     do
#     for model in 'factored_simple' 'simple'
#         do
#         for dim in 3
#             do
#             for lr in 0.01 0.005
#                 do
#                 run_name="${model}_lr_${lr}_dim_${dim}_${i}"
#                 sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py seed=${i} experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.initial_exploration_steps=${initial_exploration_steps} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#             done
#         done
#     done
# done
# model='lasso_simple'
# initial_exploration_steps=1000
# for i in {1..3}
#     do
#     for dim in 3
#         do
#         for lr in 0.01 0.005
#             do
#             run_name="${model}_lr_${lr}_targetdelta_${target_is_delta}_dim_${dim}_${i}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py seed=${i} experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.initial_exploration_steps=${initial_exploration_steps} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# planning_horizon=100
# dim=7
# lr=0.0001
# for i in {1..3}
#     do
#     for model in 'factored_simple' 'simple'
#         do
#         for lr in 0.001 0.0001
#             do
#             run_name="${model}_lr_${lr}_targetdelta_${target_is_delta}_dim_${dim}_${i}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py seed=${i} experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} overrides.planning_horizon=${planning_horizon} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.initial_exploration_steps=${initial_exploration_steps} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# overrides='pets_hypergrid'
# lr=0.001
# num_epochs_train_model=10
# target_is_delta=true
# initial_exploration_steps=50
# for i in {1..3}
#     do
#     for model in 'factored_gp' 'gaussian_process'
#         do
#         for dim in 2 3
#             do
#             for lr in 0.001 0.0001
#             do
#                 run_name="${model}_lr_${lr}_targetdelta_${target_is_delta}_dim_${dim}_${i}"
#                 sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=2048 --output="output/%J" --wrap="python3 run_scripts/train.py seed=${i} experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.initial_exploration_steps=${initial_exploration_steps} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#             done
#         done
#     done
# done


# group_name='run_hypergrid'
# overrides='pets_hypergrid'
# model='linear_regression'
# num_epochs_train_model=30
# for dim in 2 5
#     do
#     for target_is_delta in true false
#         do 
#         for lr in 0.1 0.01 0.001
#             do
#             run_name="${model}_lr_${lr}_targetdelta_${target_is_delta}_dim_${dim}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# model='gaussian_process'
# num_epochs_train_model=10
# for dim in 2 5
#     do
#     for target_is_delta in true false
#         do 
#         for lr in 0.1 0.01 0.001
#             do
#             run_name="${model}_lr_${lr}_targetdelta_${target_is_delta}_dim_${dim}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# model='simple'
# dim=5
# num_epochs_train_model=10
# for i in {1..3}
#     do
#     for target_is_delta in true false
#         do 
#         for lr in 0.01 0.005 0.001
#             do
#             run_name="${model}_lr_${lr}_targetdelta_${target_is_delta}_dim_${dim}_${i}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.env_config.grid_dim=${dim} overrides.num_epochs_train_model=${num_epochs_train_model} dynamics_model=${model} dynamics_model.model_trainer.optim_lr=${lr} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done
# group_name='art_4step_20centroid_corrected'
# overrides='pets_bikes_20centroid'
# rescale_input=true
# rescale_output=true
# #obs_postprocess_fn='obs_postprocess_pred_proba'
# for i in {1..3}
#     do
#     sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
#     sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
#     for model in 'factored_gp' 'gaussian_process' 'mixture'
#         do
#         for target_is_delta in false true
#             do
#             for obs_postprocess_fn in 'obs_postprocess_fn' 'obs_postprocess_pred_proba'
#                 do
#                 run_name="${model}_${obs_postprocess_fn}_targetdelta_${target_is_delta}_${i}"
#                 sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=2048 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#             done
#         done
#     done
# done

# group_name='test_targetisdelta_art_4step_20centroid'
# overrides='pets_bikes_20centroid'
# rescale_input=true
# for i in {1..5}
#     do
#     sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
#     sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
#     for rescale_output in true false
#         do
#         for target_is_delta in true false
#             do
#             run_name="rescalein_${rescale_input}_rescaleout_${rescale_output}_targetdelta_${target_is_delta}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=3072 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# group_name='test_targetisdelta_real_4step_20centroid'
# for i in {1..5}
#     do
#     sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
#     sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
#     for rescale_output in true false
#         do
#         for target_is_delta in true false
#             do
#             run_name="rescalein_${rescale_input}_rescaleout_${rescale_output}_targetdelta_${target_is_delta}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=3072 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# group_name='artificial_4step_5centroid_test'
# overrides='pets_bikes_5centroid'
# sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
# sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
# for rescale_input in true false
#     do
#     for rescale_output in true false
#         do
#         for target_is_delta in true false
#             do
#             run_name="rescalein_${rescale_input}_rescaleout_${rescale_output}_targetdelta_${target_is_delta}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#         done
#     done
# done

# group_name='artificial_4step_20centroid'
# overrides='pets_bikes_20centroid'
# sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
# sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
# for model in "gaussian_process" "factored_gp"
#     do
#     for lr in 0.1 0.01
#         do
#         run_name="${model}_lr_${lr}"
#         sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} dynamics_model=${model} dynamics_model.model_lr=${lr} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#     done
# done

# group_name='real_4step_20centroid'

# trips_data="src/env/bikes_data/all_trips_LouVelo_merged.csv"
# weather_data="src/env/bikes_data/weather_data.csv"
# sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} group_name=${group_name}"
# sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} group_name=${group_name}"
# for model in "gaussian_process" "factored_gp"
#     do
#     for lr in 0.1 0.01
#         do
#         run_name="${model}_lr_${lr}"
#         sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} dynamics_model=${model} dynamics_model.model_lr=${lr} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} overrides.model_wrapper.model_input_obs_key='['bikes_distr','demands','time_counter']' experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
#     done
# done
