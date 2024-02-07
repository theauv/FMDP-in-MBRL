#!/bin/bash

module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate
pip install -e .

sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true agent='random' additional_run_name='multistep_artificial'"
sbatch -n 1 --cpus-per-task=2 --time=24:00:00 --mem-per-cpu=1024 --output="output/%J" --wrap="python3 run_scripts/benchmark_run.py with_tracking=true agent='good_heuristic' additional_run_name='multistep_artificial'"