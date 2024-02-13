#!/bin/bash

module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .

group_name='test_nonfactored_targetisdelta_art_4step_5centroid'
overrides='pets_bikes_5centroid'
rescale_input=true
model='gaussian_process'
for i in {1..3}
    do
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
    for rescale_output in true false
        do
        for target_is_delta in true false
            do
            run_name="rescalein_${rescale_input}_rescaleout_${rescale_output}_targetdelta_${target_is_delta}"
            sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} model_dynamics=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
        done
    done
done

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
