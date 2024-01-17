#!/bin/bash


module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .

# batch
# for i in {1..5}
#     do

# done


group_name="euler_bikes_test"

for station_dependencies in "src/env/bikes_data/factors_radius_0-4.npy" "src/env/bikes_data/factors_radius_1-2.npy" "src/env/bikes_data/factors_radius_2.npy" "src/env/bikes_data/factors_radius_3.npy"
    do
    for layers in 2 4 6
        do
        for hid_size in 100 200 400
            do
            radius=${radius##*/}
            radius=${radius%%.*}
            run_name="factored_${radius}_layers_${layers}_hidsize_${hid_size}"
            sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --mail-type=END --wrap="python3 run_scripts/train.py overrides=pets_bikes overrides.env_config.station_dependencies=${station_dependencies} dynamics_model=factored_simple dynamics_model.model.num_layers=${layers} dynamics_model.model.hid_size=${hid_size} experiment.run_configs.group=${group_name} experiment.run_configs.name=${run_name}"
        done
    done
done
