# Factored Markov Decision Processes (FMDP) in Model-Based Reinforcement Learning (MBRL)

This repository contains all the code associated to Théau Vannier's master thesis, student in Computational Sciences and Engineering at EPFL, doing a master's thesis at ETHZ.

## Background


## Getting started

The Model-based RL framework of this project is mainly based on the mbrl-lib (!cite!)
mbrl requires Python 3.8+ library and PyTorch (>= 1.7). To install the latest stable version, run

```
pip install mbrl
```

Add other library (wandb, torch, omegaconf, hydra etc ? Or requirements.txt)

## Framework

# Configs

The parameters of each building block of the project are gathered in yaml files in the "configs" directory. As a code user you only need to make changes in there.
- action_optimizer: ??
- algorithm: contains everything about the agent and the algorithm used to optimize the policy
- dynamics_model: contains everything about the model learning the dynamic of the envionment
- experiment: contains everything about the api dealing with the logs and plots (e.g. wandb). One yaml file per 
- overrides: contains everything about the environment (there is optionally an env_config). One yaml file per environment.
- main: gather everything, choose a yaml file for each building block described above.



## Use
