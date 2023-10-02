import omegaconf
import torch
import hydra
import numpy as np

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet

from env.env_handler import EnvironmentHandler


@hydra.main(config_path="configs", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = EnvironmentHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = EnvironmentHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)


if __name__ == "__main__":
    run()
