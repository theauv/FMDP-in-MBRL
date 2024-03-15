#!/bin/bash

group_name=""

python3 run_scripts/benchmark_run.py agent="random" additional_run_name="1step"
python3 run_scripts/benchmark_run.py agent="good_heuristic" additional_run_name="1step"
