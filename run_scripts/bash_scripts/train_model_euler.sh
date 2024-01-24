#!/bin/bash
module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .


for model in "gaussian_process" "simple"
    do
    for learned_reward in false true
        do
        for metric in MSE R2
            do
            run_name="test_${model}_LRew_${learned_reward}_${metric}"
            sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=2048 --output="output/%J" --mail-type=END --wrap="python3 run_scripts/train_model.py overrides=pets_bikes debug_mode=false learned_rewards=${learned_reward} dynamics_model=${model} dynamics_model.model.eval_metric=${metric} run_name=${run_name}"
        done
    done
done

# for station_dependencies in "src/env/bikes_data/factors_radius_1-2.npy" "src/env/bikes_data/factors_radius_3.npy"
#     do
#     for layers in 4 6
#         do
#         for hid_size in 200 400
#             do
#             radius=${radius##*/}
#             radius=${radius%%.*}
#             run_name="gaussianMLP_${radius}_layers_${layers}_hidsize_${hid_size}"
#             sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --mail-type=END --wrap="python3 run_scripts/train_model.py overrides=pets_bikes overrides.env_config.station_dependencies=${station_dependencies} dynamics_model=gaussian_mlp_ensemble dynamics_model.model.num_layers=${layers} dynamics_model.model.hid_size=${hid_size} run_name=${run_name}"
#         done
#     done
# done
