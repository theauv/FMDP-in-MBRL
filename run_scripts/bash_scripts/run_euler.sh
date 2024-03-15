#!/bin/bash

# bash
# module load eth_proxy
# module load gcc/8.2.0 python/3.9.9
# conda activate pdm-env
# pip install -e .

#group_name='real_4step'
group_name='art_4step'
overrides='pets_bikes'
rescale_input=true
target_is_delta=true
trips_data="src/env/bikes_data/all_trips_LouVelo_merged.csv"
weather_data="src/env/bikes_data/weather_data.csv"
initial_exploration_steps=50
obs_postprocess_fn='obs_postprocess_fn'
for i in {1..3}
    do
    #sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} agent='random' group_name=${group_name}"
    #sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} agent='good_heuristic' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
    for model in 'factored_gp' 'gaussian_process'
        do
        for rescale_output in true false
            do
            run_name="${model}_${obs_postprocess_fn}_rescale_out_${rescale_output}_${i}"
            sbatch -n 1 --cpus-per-task=4 --time=48:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} overrides.initial_exploration_steps=${initial_exploration_steps} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
            #sbatch -n 1 --cpus-per-task=4 --time=48:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} overrides.initial_exploration_steps=${initial_exploration_steps} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
        done
    done
done

#group_name='real_4step_20centroid'
group_name='art_4step_20centroid'
overrides='pets_bikes_20centroid'
trips_data="src/env/bikes_data/all_trips_LouVelo_merged.csv"
weather_data="src/env/bikes_data/weather_data.csv"
initial_exploration_steps=50
for i in {1..3}
    do
    #sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} agent='random' group_name=${group_name}"
    #sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} agent='good_heuristic' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='random' group_name=${group_name}"
    sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=512 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true overrides=${overrides} agent='good_heuristic' group_name=${group_name}"
    for model in 'factored_gp' 'gaussian_process'
        do
        for rescale_output in true false
            do
            run_name="${model}_${obs_postprocess_fn}_rescale_out_${rescale_output}_${i}"
            sbatch -n 1 --cpus-per-task=4 --time=48:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} overrides.initial_exploration_steps=${initial_exploration_steps} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
            #sbatch -n 1 --cpus-per-task=4 --time=48:00:00 --mem-per-cpu=4000 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=${overrides} overrides.obs_postprocess_fn=${obs_postprocess_fn} overrides.initial_exploration_steps=${initial_exploration_steps} overrides.env_config.past_trip_data=${trips_data} overrides.env_config.weather_data=${weather_data} dynamics_model=${model} algorithm.rescale_input=${rescale_input} algorithm.rescale_output=${rescale_output} algorithm.target_is_delta=${target_is_delta} experiment.run_configs.name=${run_name} experiment.run_configs.group=${group_name}"
        done
    done
done
