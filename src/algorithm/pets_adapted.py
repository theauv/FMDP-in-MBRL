# Copied and adpat from mbrl-lib github:
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Dict

import gymnasium as gym
from gymnasium import spaces
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

from src.callbacks.wandb_callbacks import CallbackWandb
from src.callbacks.constants import RESULTS_LOG_NAME, EVAL_LOG_FORMAT
from src.util.common_overriden import (
    create_one_dim_tr_model_overriden,
    step_env_and_add_to_buffer_overriden,
    train_model_and_save_model_and_data_overriden,
)


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    """Training pipeline using a Probabilistic Ensemble Trajectory Sampling (PETS) approach
    Chua, Kurtland, et al. "Deep reinforcement learning in a handful of trials using probabilistic 
    dynamics models." Advances in neural information processing systems 31 (2018).

    This function is adapted from the original mbrl.algorithms.pets one to handle callbacks and 
    possible future changes.

    :param termination_fn: termination function associated to the given environment
    :param reward_fn: reward function associated to the given environment
    :param cfg: configuration file (see configs directory)
    :param silent: no logs if True , defaults to False
    :param work_dir: directory where results will be saved, defaults to None
    :return: max total reward achieved
    """

    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green")

    # -------- Create and populate initial env dataset --------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    replay_buffer.save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_path = cfg.overrides.get("model_path", None)

    dynamics_model = create_one_dim_tr_model_overriden(
        cfg, env, obs_shape, act_shape, model_path
    )
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = hydra.utils.instantiate(
        cfg.dynamics_model.model_trainer,
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    # ---------------------------------------------------------
    # ----------------- Callbacks -----------------------------
    callbacks = CallbackWandb(
        cfg.experiment.with_tracking,
        max_traj_iterations=cfg.overrides.cem_num_iters,
        model_out_size=dynamics_model.model.out_size,
        plot_local=cfg.experiment.plot_local,
    )

    callbacks.env_callback(env)

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    current_trial = 0
    max_total_reward = -np.inf

    while (
        env_steps < cfg.overrides.num_steps
        and current_trial < cfg.overrides.num_episodes
    ):
        obs, _ = env.reset()
        agent.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps_trial = 0

        # Make 1 episode
        while not terminated and not truncated:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:
                train_model_and_save_model_and_data_overriden(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                    callback=callbacks.model_train_callback,
                    callback_sparsity=callbacks.model_sparsity,
                )
                if env_steps == 0 and hasattr(dynamics_model.model, "factors"):
                    callbacks.model_dbn(dynamics_model.model.factors)

            # --- Doing env step using the agent and adding to model dataset ---
            callbacks.env_step += 1
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = step_env_and_add_to_buffer_overriden(  # locally overriden to handle callbacks
                env,
                obs,
                agent,
                {},
                replay_buffer,
                optimizer_callback=callbacks.trajectory_optimizer_callback,
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1

            if debug_mode:
                print(f"Step {env_steps}: Reward {reward:.3f}.")

        current_trial += 1
        if logger is not None:
            logger.log_data(
                RESULTS_LOG_NAME,
                {
                    "trial": current_trial,
                    "env_step": env_steps,
                    "episode_reward": total_reward,
                },
            )
            callbacks.agent_callback(current_trial, steps_trial, total_reward)
        if debug_mode:
            print(f"Trial: {current_trial }, reward: {total_reward}.")

        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
