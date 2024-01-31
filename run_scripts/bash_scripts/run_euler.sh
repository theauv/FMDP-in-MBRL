#!/bin/bash


module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .

# batch
# for i in {1..5}
#     do

# done


# group_name="euler_bikes_test"

# for station_dependencies in "src/env/bikes_data/factors_radius_0-4.npy" "src/env/bikes_data/factors_radius_1-2.npy" "src/env/bikes_data/factors_radius_2.npy" "src/env/bikes_data/factors_radius_3.npy"
#     do
#     for layers in 2 4 6
#         do
#         for hid_size in 100 200 400
#             do
#             radius=${radius##*/}
#             radius=${radius%%.*}
#             run_name="factored_${radius}_layers_${layers}_hidsize_${hid_size}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --mail-type=END --wrap="python3 run_scripts/train.py overrides=pets_bikes overrides.env_config.station_dependencies=${station_dependencies} dynamics_model=factored_simple dynamics_model.model.num_layers=${layers} dynamics_model.model.hid_size=${hid_size} experiment.run_configs.group=${group_name} experiment.run_configs.name=${run_name}"
#         done
#     done
# done

group_name="reward_only_1step"

# for station_dependencies in "src/env/bikes_data/factors_radius_1-2.npy" "src/env/bikes_data/factors_radius_3.npy"
#     do
#     for model in "linear_regression" "simple"
#         do
#         radius=${station_dependencies##*/}
#         radius=${radius%%.*}
#         run_name="${model}_${radius}"
#         sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py overrides=pets_bikes overrides.env_config.station_dependencies=${station_dependencies} dynamics_model=${model} experiment.with_tracking=true experiment.run_configs.group=${group_name} experiment.run_configs.name=${run_name}"
#     done
# done

group_name="reward_only_1step_tests"

sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="linear_regression" experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="linear_regression" overrides.freq_train_model=20 experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="linear_regression" overrides.num_epochs_train_model=100 experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="linear_regression" overrides.freq_train_model=20 overrides.num_epochs_train_model=100 experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="simple" experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="simple" overrides.freq_train_model=20 experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="simple" overrides.num_epochs_train_model=100 experiment.run_configs.name=${run_name}"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py experiment.with_tracking=true overrides=pets_bikes experiment.run_configs.group=${group_name} dynamics_model="simple" overrides.freq_train_model=20 overrides.num_epochs_train_model=100 experiment.run_configs.name=${run_name}"


# for cem_num_iters in 20 40 60
#     do
#     for cem_alpha in 0.001 0.1
#         do
#         run_name="iters_${cem_num_iters}_alpha_${cem_alpha}"
#         sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train.py overrides=pets_bikes overrides.cem_num_iters=${cem_num_iters} overrides.cem_alpha=${cem_alpha} experiment.with_tracking=true experiment.run_configs.group=${group_name} experiment.run_configs.name=${run_name}"
#     done
# done