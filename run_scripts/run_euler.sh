#!/bin/bash


module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .

# for i in {1..5}
#     do

# done

group_name="euler_bikes_test"

sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --mail-type=END --wrap="python3 run_scripts/train.py overrides=pets_bikes experiment.run_configs.group=${group_name}"

