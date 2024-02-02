import git
import gymnasium as gym
import hydra
import logging
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import omegaconf
from pathlib import Path
import shutil
from time import sleep
import wandb

from mbrl.planning.core import Agent, RandomAgent

from src.env.bikes import Bikes
from src.env.hypergrid import ContinuousHyperGrid
from src.callbacks.wandb_callbacks import CallbackWandb
from src.agent.heuristic import StubbornAgent, GoodBikesHeuristic


def run_agent_in_env(
    env: gym.Env, agent: Agent, num_steps: int, callbacks: CallbackWandb
):
    # create env and random seed
    observation, info = env.reset(seed=42)
    if callbacks is not None:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        callbacks.env_callback(env)

    # create env and random seed
    observation, info = env.reset(seed=42)

    all_n_steps = []
    all_rewards = []
    # Debug
    all_n_feasible_trips = []
    all_ratio_feasible_trips = []

    n_steps = 0
    rewards = 0
    for env_step in range(num_steps):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"Total env step: {env_step} Episode: {len(all_n_steps)} Current episode step: {n_steps} "
            f"Reward: {reward}"
        )
        n_steps += 1
        rewards += reward

        if callbacks is not None:
            callbacks.track_each_step(env_step, reward)
        elif env.render_mode == "human":
            input()
            #sleep(1)

        if terminated or truncated:
            all_n_steps.append(n_steps)
            all_rewards.append(rewards)
            # if callbacks is not None:
            #     callbacks.agent_callback(len(all_n_steps), n_steps, rewards)
            observation, info = env.reset()
            n_steps = 0
            rewards = 0
    env.close()
    print(f"Mean reward: {np.mean(all_rewards)}")
    print(f"Mean feasible trips: {np.mean(all_n_feasible_trips)}")
    print(f"Mean feasible ratio trips: {np.mean(all_ratio_feasible_trips)}")

    if callbacks is None:
        matplotlib.use("TkAgg")
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(all_n_steps)
        axs[1].plot(all_rewards)
        axs[0].set_xlabel("Episode")
        axs[1].set_xlabel("Episode")
        axs[0].set_ylabel("Number of steps")
        axs[1].set_ylabel("Total reward")
        plt.show()


@hydra.main(config_path="../configs", config_name="benchmarking")
def run(cfg: omegaconf.DictConfig):
    if cfg.with_tracking:
        cfg.render_mode = None

    # Choose the environment
    if cfg.env_name == "hypergrid":
        env = ContinuousHyperGrid(
            env_config=cfg.env_config, render_mode=cfg.render_mode
        )
    elif cfg.env_name == "bikes":
        env = Bikes(env_config=cfg.env_config, render_mode=cfg.render_mode)
    else:
        raise ValueError(f"No environment called {cfg.env_name} implemented")

    # Choose the benchmark agent
    if cfg.agent == "random":
        agent = RandomAgent(env)
    elif cfg.agent == "stubborn":
        # TODO: add a action parameter for stubborn in configs
        agent = StubbornAgent(env, action=None)
    elif cfg.agent == "good_heuristic":
        agent = GoodBikesHeuristic(cfg.env_config, env)
    else:
        raise ValueError(f"No benchmark agent called {cfg.agent} implemented (yet)")

    callbacks = None
    if cfg.with_tracking:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        group_name = cfg.get("group_name", f"Benchmarking_{cfg.env_name}")
        run_name = f"{cfg.agent}"
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
            "tags": [cfg.agent],
        }

        init_run_kwargs = wandb_config
        init_run_kwargs["config"] = omegaconf.OmegaConf.to_container(cfg, resolve=True)
        wandb.init(**init_run_kwargs)
        callbacks = CallbackWandb(cfg.with_tracking)
    run_agent_in_env(env, agent, cfg.num_steps, callbacks)


if __name__ == "__main__":
    run()
