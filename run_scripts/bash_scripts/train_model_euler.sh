#!/bin/bash
# module load eth_proxy
# module load gcc/8.2.0 python/3.9.9
# source euler-pdm-env/bin/activate
# pip install -e .

for model in "linear_regression" "ffnn"
    do
    for learned_reward in false true
        do
        for dataset_size in 500 2000
            do
            run_name="test_${model}_LRew_${learned_reward}_datasize_${dataset_size}"
            dataset_folder_name="datasets/temporary/${run_name}"
            sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/train_model.py overrides=pets_bikes debug_mode=false learned_rewards=${learned_reward} dataset_size=${dataset_size} dataset_folder_name=${dataset_folder_name} dynamics_model=${model} run_name=${run_name}"
        done
    done
done
