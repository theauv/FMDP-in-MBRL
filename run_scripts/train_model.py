import git
import gymnasium as gym
import hydra
import logging
import numpy as np
import omegaconf
import os
import torch
import wandb

import mbrl
from mbrl.util.common import create_replay_buffer, rollout_agent_trajectories

from src.callbacks.wandb_callbacks import CallbackWandb
from src.env.bikes import Bikes
from src.env.env_handler import HandMadeEnvHandler
from src.util.common_overriden import train_model_and_save_model_and_data_overriden, create_one_dim_tr_model_overriden
from src.util.util import get_run_kwargs


def train_model(cfg, env: gym.Env):

    # -------- Initialization --------
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    env_is_bikes = False
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    if isinstance(base_env, Bikes):
        env_is_bikes = True
        base_env.set_next_day_method("random")
    
    # -------- Create and populate initial env dataset --------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32

    #TODO: add a load dir (already exists, but set it)
    replay_buffer = create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    rollout_agent_trajectories(
        env,
        cfg.dataset_size,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    if env_is_bikes:
        base_env.set_next_day_method(cfg.overrides.env_config.next_day_method)

    dir_path = f"datasets/{base_env.__class__.__name__}/{cfg.dataset_size}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 
        replay_buffer.save(dir_path)

    # -------- Create model and model trainer --------
    model_path = cfg.overrides.get("model_path", None)

    dynamics_model = create_one_dim_tr_model_overriden(
        cfg, env, obs_shape, act_shape, model_path
    )
    model_trainer = hydra.utils.instantiate(
        cfg.dynamics_model.model_trainer,
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None,
    )

    # ----------------- Callbacks -----------------------------
    callbacks = CallbackWandb(
        cfg.experiment.with_tracking,
        max_traj_iterations=cfg.overrides.cem_num_iters,
        model_out_size=dynamics_model.model.out_size,
        plot_local=cfg.experiment.plot_local,
        centroid_coords=env.centroid_coords,
    )
    if hasattr(dynamics_model.model, "factors"):
        callbacks.model_dbn(dynamics_model.model.factors)

    # ------------- Train -------------
    print("Training start")
    train_model_and_save_model_and_data_overriden(
        dynamics_model,
        model_trainer,
        cfg,
        replay_buffer,
        work_dir=None,
        callback=callbacks.model_train_callback,
        callback_sparsity=callbacks.model_sparsity,
    )
    print("Training end")


@hydra.main(config_path="../configs", config_name="training_model")
def run(cfg: omegaconf.DictConfig):

    # set-up run and api
    if cfg.debug_mode:
        cfg.experiment.with_tracking=False
    else:
        cfg.experiment.with_tracking=True
    if cfg.experiment.with_tracking:
        cfg.overrides.render_mode = None
        if cfg.experiment.api_name == "wandb":
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            group_name = cfg.get("group_name", f"Train_model")
            run_name = f"{sha}"
            if cfg.additional_run_name:
                run_name += f"_{cfg.additional_run_name}"
                logging.basicConfig(
                    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO,
                )
            wandb_config = {
                "project": "hucrl_fmdp",
                "group": group_name,
                "name": run_name,
                "settings": None,
            }
            init_run_kwargs = wandb_config
            init_run_kwargs["config"] = omegaconf.OmegaConf.to_container(cfg, resolve=True)
            wandb.init(**init_run_kwargs)
        else:
            raise ValueError("Unsupported API")
    
    # create env and random seed
    env, term_fn, reward_fn = HandMadeEnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_model(cfg, env)


if __name__ == "__main__":
    run()
