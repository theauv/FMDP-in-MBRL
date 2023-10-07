import argparse
import pathlib
from typing import List, Optional, Tuple
from time import sleep
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

import mbrl
from mbrl.diagnostics.visualize_model_preds import Visualizer
import mbrl.models
import mbrl.planning
import mbrl.util.common

from src.env.env_handler import get_handler


VisData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


#TODO: Make sure that agent+model correctly loaded !!!!
#Seems to be good 


class AdaptedVisualizer(Visualizer):
    def __init__(
        self,
        lookahead: int,
        results_dir: str,
        random_agent: bool = False,
        num_steps: Optional[int] = None,
        num_model_samples: int = 1,
        model_subdir: Optional[str] = None,
    ):
        self.lookahead = lookahead
        self.results_path = pathlib.Path(results_dir)
        self.model_path = self.results_path
        self.vis_path = self.results_path / "diagnostics"
        if model_subdir:
            self.model_path /= model_subdir
            # If model subdir is child of diagnostics, remove "diagnostics" before
            # appending to vis_path. This can happen, for example, if Finetuner
            # generated this model with a model_subdir
            if "diagnostics" in model_subdir:
                model_subdir = pathlib.Path(model_subdir).name
            self.vis_path /= model_subdir
        pathlib.Path.mkdir(self.vis_path, parents=True, exist_ok=True)

        self.num_model_samples = num_model_samples
        self.num_steps = num_steps

        self.cfg = mbrl.util.common.load_hydra_cfg(self.results_path)
        self.cfg.overrides.render_mode = "human"

        #Only changed line
        self.handler = get_handler(self.cfg)
        self.env, term_fn, reward_fn = self.handler.make_env(self.cfg)

        self.reward_fn = reward_fn

        self.dynamics_model = mbrl.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=self.model_path,
        )
        self.model_env = mbrl.models.ModelEnv(
            self.env,
            self.dynamics_model,
            term_fn,
            reward_fn,
            generator=torch.Generator(self.dynamics_model.device),
        )

        #Instanciate the agent
        self.agent: mbrl.planning.Agent
        if random_agent:
            self.agent = mbrl.planning.RandomAgent(self.env)
        else:
            agent_dir = self.results_path
            agent_cfg = mbrl.util.common.load_hydra_cfg(agent_dir)
            if (
                agent_cfg.algorithm.agent._target_
                == "mbrl.planning.TrajectoryOptimizerAgent"
            ):
                print("Agent uses TrajectoryOptimizer")
                agent_cfg.algorithm.agent.planning_horizon = lookahead
                self.agent = mbrl.planning.create_trajectory_optim_agent_for_model(
                    self.model_env,
                    agent_cfg.algorithm.agent,
                    num_particles=agent_cfg.algorithm.num_particles,
                )
            else:
                self.agent = mbrl.planning.load_agent(agent_dir, self.env)

        self.fig = None
        self.axs: List[plt.Axes] = []
        self.lines: List[plt.Line2D] = []
        self.writer = animation.FFMpegWriter(
            fps=15, metadata=dict(artist="Me"), bitrate=1800
        )

        # The total reward obtained while building the visualizationn
        self.total_reward = 0

    def simple_run(self):

        # create env and random seed
        observation, info = self.env.reset(seed=42)

        all_n_steps = []
        all_rewards = []

        n_steps = 0
        rewards = 0
        for env_step in range(1000):
            print(f"Total env step: {env_step} Episode: {len(all_n_steps)} Current episode step: {n_steps}")
            action = self.agent.act(observation)  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = self.env.step(action)

            n_steps += 1
            rewards += reward

            if terminated or truncated:
                all_n_steps.append(n_steps)
                all_rewards.append(rewards)
                sleep(0.5)
                observation, info = self.env.reset()
                n_steps = 0
                rewards = 0
        self.env.close()

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
        "--experiments_dir",
        type=str,
        default=None,
        help="The directory where the original experiment was run.",
    )
    parser.add_argument("--lookahead", type=int, default=25)
    parser.add_argument('--random_agent', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument(
        "--model_subdir",
        type=str,
        default=None,
        help="Can be used to point to models generated by other diagnostics tools.",
    )
    parser.add_argument(
        "--num_model_samples",
        type=int,
        default=35,
        help="Number of samples from the model, to visualize uncertainty.",
    )
    args = parser.parse_args()

    visualizer = AdaptedVisualizer(
        lookahead=args.lookahead,
        results_dir=args.experiments_dir,
        random_agent=args.random_agent,
        num_steps=args.num_steps,
        num_model_samples=args.num_model_samples,
        model_subdir=args.model_subdir,
    )

    visualizer.simple_run()