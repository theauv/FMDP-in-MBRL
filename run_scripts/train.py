import wandb
import torch
import omegaconf
import hydra
import numpy as np
import warnings

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet

import src.algorithm.pets_adapted as pets_adapted
from src.env.env_handler import HandMadeEnvHandler
from src.util.util import get_run_kwargs


@hydra.main(config_path="../configs", config_name="main")
def run(cfg: omegaconf.DictConfig):
    # set-up run and api
    if cfg.experiment.with_tracking:
        cfg.overrides.render_mode = None
        if cfg.experiment.api_name == "wandb":
            # start a new wandb run to track this script
            wandb.init(**get_run_kwargs(cfg))
        else:
            raise ValueError("Unsupported API")

    # create env and random seed
    env, term_fn, reward_fn = HandMadeEnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # train the agent and model with the given algorithm
    if cfg.algorithm.name == "pets_adapted":
        return pets_adapted.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "pets":
        warnings.warn("Might not be supported yet in the scope of this project")
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        warnings.warn("Might not be supported yet in the scope of this project")
        test_env, *_ = HandMadeEnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        warnings.warn("Might not be supported yet in the scope of this project")
        return planet.train(env, cfg)


if __name__ == "__main__":
    run()
