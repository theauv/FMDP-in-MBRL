#!/bin/bash

group_name=""

python3 run_scripts/benchmark_run.py agent="random" additional_run_name="1step"
python3 run_scripts/benchmark_run.py agent="good_heuristic" additional_run_name="1step"

#python -m run_scripts.benchmark_run agent="random" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='random' overrides.env_config.bikes_per_truck=5 overrides.env_config.start_walk_dist_max=1 additional_run_name="trucks_5_bikes_5_walk_1_trip05"
#python -m run_scripts.benchmark_run agent="random" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='random' overrides.env_config.bikes_per_truck=10 overrides.env_config.start_walk_dist_max=1 additional_run_name="trucks_5_bikes_10_walk_1_trip05"

# python -m run_scripts.benchmark_run agent="good_heuristic" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='random' overrides.env_config.bikes_per_truck=10 overrides.env_config.start_walk_dist_max=1 additional_run_name="trucks_5_bikes_10_walk_1_trip05"
# python -m run_scripts.benchmark_run agent="good_heuristic" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='random' overrides.env_config.bikes_per_truck=10 overrides.env_config.start_walk_dist_max=0.5 additional_run_name="trucks_5_bikes_10_walk_05_trip05"
# python -m run_scripts.benchmark_run agent="good_heuristic" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='random' overrides.env_config.bikes_per_truck=10 overrides.env_config.start_walk_dist_max=0.3 additional_run_name="trucks_5_bikes_10_walk_03_trip05"
# python -m run_scripts.benchmark_run agent="good_heuristic" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='random' overrides.env_config.bikes_per_truck=10 overrides.env_config.start_walk_dist_max=0.2 additional_run_name="trucks_5_bikes_10_walk_02_trip05"

# python -m run_scripts.benchmark_run agent="random" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='sequential' overrides.env_config.bikes_per_truck=5 overrides.env_config.start_walk_dist_max=1 additional_run_name="trucks_5_bikes_5_walk_1_trip05"
# python -m run_scripts.benchmark_run agent="random" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='sequential' overrides.env_config.bikes_per_truck=5 overrides.env_config.start_walk_dist_max=0.5 additional_run_name="trucks_5_bikes_5_walk_05_trip05"
# python -m run_scripts.benchmark_run agent="random" overrides.env_config.num_trucks=5 overrides.env_config.initial_distribution='zeros' overrides.env_config.next_day_method='sequential' overrides.env_config.bikes_per_truck=5 overrides.env_config.start_walk_dist_max=0.2 additional_run_name="trucks_5_bikes_5_walk_02_trip05"

# python -m run_scripts.benchmark_run agent="random" overrides.env_config.next_day_method='random' overrides.env_config.trip_duration=0.2 additional_run_name="trucks_5_bikes_5_walk_02_trip02"
# python -m run_scripts.benchmark_run agent="random" overrides.env_config.next_day_method='random' overrides.env_config.trip_duration=1 additional_run_name="trucks_5_bikes_5_walk_02_trip3"
# python -m run_scripts.benchmark_run agent="random" overrides.env_config.next_day_method='random' overrides.env_config.trip_duration=3 additional_run_name="trucks_5_bikes_5_walk_02_trip3"

# python -m run_scripts.benchmark_run agent="random" overrides.env_config.next_day_method='sequential' overrides.env_config.end_walk_dist_max=4 additional_run_name="end_walk_dist_4"
# python -m run_scripts.benchmark_run agent="random" overrides.env_config.next_day_method='sequential' overrides.env_config.end_walk_dist_max=0.5 additional_run_name="end_walk_dist_05"
# python -m run_scripts.benchmark_run agent="random" overrides.env_config.next_day_method='sequential' overrides.env_config.end_walk_dist_max=1000 additional_run_name="end_walk_dist_1000"
