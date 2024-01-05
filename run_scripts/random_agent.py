import argparse
import gymnasium as gym
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
from time import sleep

from src.env.bikes import Bikes
from src.env.hypergrid import ContinuousHyperGrid


def random_agent(env: gym.Env, num_steps: int):
    # create env and random seed
    observation, info = env.reset(seed=42)

    all_n_steps = []
    all_rewards = []

    n_steps = 0
    rewards = 0
    for env_step in range(num_steps):
        print(
            f"Total env step: {env_step} Episode: {len(all_n_steps)} Current episode step: {n_steps}"
        )
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        n_steps += 1
        rewards += reward
        sleep(1)

        if terminated or truncated:
            all_n_steps.append(n_steps)
            all_rewards.append(rewards)
            observation, info = env.reset()
            n_steps = 0
            rewards = 0
    env.close()

    print(f"Mean reward: {np.mean(all_rewards)}")

    matplotlib.use("TkAgg")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(all_n_steps)
    axs[1].plot(all_rewards)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[0].set_ylabel("Number of steps")
    axs[1].set_ylabel("Total reward")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="hypergrid",
        help="The name of the gym environment",
    )
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="Render mode you want to use when runnind the random agent",
    )
    args = parser.parse_args()

    if args.env_name == "hypergrid":
        # TODO change this bad hard coding
        env_config = {
            "step_penalty": -1,
            "grid_dim": 2,
            "grid_size": 5.0,
            "size_end_box": 1.0,
            "step_size": 1.0,
            "n_obstacles": 5,
            "size_obstacles": 1,
        }
        env_config = DictConfig(env_config)

        env = ContinuousHyperGrid(env_config=env_config, render_mode=args.render_mode)
    elif args.env_name == "bikes":
        env_config = {
            "num_trucks": 5,
            "action_per_day": 8,
            "next_day_method": "sequential",
            "initial_distribution": "zeros",
            "bikes_per_truck": 15,
            "walk_distance_max": 1.0,
            "past_trip_data": "src/env/bikes_data/all_trips_LouVelo_merged.csv",
            "weather_data": "src/env/bikes_data/weather_data.csv",
            "centroids_coord": "src/env/bikes_data/LouVelo_centroids_coords.npy",
        }
        env_config = DictConfig(env_config)

        env = Bikes(env_config=env_config, render_mode=args.render_mode)
    else:
        raise ValueError(f"No environment called {args.env_name} implemented")

    random_agent(env, args.num_steps)
