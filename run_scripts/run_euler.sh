git pull
module load eth_proxy
module load gcc/8.2.0 python/3.9.9
source euler-pdm-env/bin/activate

# for i in {1..5}
#     do

# done

group_name="euler_bikes_test"

bsub -n 2 -W 24:00 -N -R "rusage[mem=1024]" 'python -m run_scripts.train overrides=pets_bikes experiment.run_configs.group=${group_name}'
