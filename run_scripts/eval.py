import argparse
import pathlib
import pygame
from random import uniform
from time import sleep
from typing import List, Optional, Tuple, Dict, Any

import hydra
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.functional import F

import mbrl
from mbrl.diagnostics.visualize_model_preds import Visualizer
import mbrl.models
import mbrl.planning
import mbrl.util.common

from src.env.bikes import Bikes
from src.env.constants import *
from src.env.hypergrid import ContinuousHyperGrid
from src.env.env_handler import get_handler
from src.env.constants import BLACK
from src.util.util import get_weights_model
from src.util.common_overriden import (
    create_one_dim_tr_model_overriden,
    create_overriden_replay_buffer,
)
from src.model.gaussian_process import MultiOutputGP

VisData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class PerfectHypergridModel(mbrl.models.Model):
    def __init__(self, device, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        # HARD CODED !!!
        self.in_size = 4
        self.out_size = 2

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        input_dim = x.shape[-1]
        assert input_dim % 2 == 0
        return x[..., input_dim // 2 :]  # x[..., :input_dim//2]+x[..., input_dim//2:]

    def loss(
        self,
        model_in,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        pred_out = self.forward(model_in)
        return F.mse_loss(pred_out, target, reduction="none").mean(-1).mean(), {}

    def eval_score(
        self, model_in, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_output = self.forward(model_in)
            return F.mse_loss(pred_output, target, reduction="none").unsqueeze(0), {}

    def update(
        self,
        model_in,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        loss, meta = self.loss(model_in, target)
        return loss.item(), meta

    def reset_1d(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        assert rng is not None
        propagation_indices = None
        return {"obs": obs, "propagation_indices": propagation_indices}

    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        return (self.forward(model_input), model_state)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class AdaptedVisualizer(Visualizer):
    def __init__(
        self,
        lookahead: int,
        results_dir: str,
        random_agent: Optional[bool] = False,
        num_steps: Optional[int] = None,
        num_model_samples: int = 1,
        model_subdir: Optional[str] = None,
        render: bool = False,
        use_perfect_hypergrid_model: bool = False,
        use_untrained_model: bool = False,
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
        if render:
            self.cfg.overrides.render_mode = "human"
        else:
            self.cfg.overrides.render_mode = "rgb_array"

        # Only changed line
        self.handler = get_handler(self.cfg)
        self.env, term_fn, reward_fn = self.handler.make_env(self.cfg)

        self.reward_fn = reward_fn

        if use_perfect_hypergrid_model:
            model = PerfectHypergridModel(device="cpu")
            name_obs_process_fn = self.cfg.overrides.get("obs_process_fn", None)
            if name_obs_process_fn:
                obs_process_fn = hydra.utils.get_method(
                    self.cfg.overrides.obs_process_fn
                )
            else:
                obs_process_fn = None
            self.dynamics_model = hydra.utils.instantiate(
                self.cfg.overrides.model_wrapper,
                model,
                target_is_delta=self.cfg.algorithm.target_is_delta,
                normalize=self.cfg.algorithm.normalize,
                normalize_double_precision=self.cfg.algorithm.get(
                    "normalize_double_precision", False
                ),
                learned_rewards=self.cfg.algorithm.learned_rewards,
                obs_process_fn=obs_process_fn,
                no_delta_list=self.cfg.overrides.get("no_delta_list", None),
                num_elites=self.cfg.overrides.get("num_elites", None),
            )
            # Overrides
            self.cfg.overrides.model_batch_size = self.cfg.dynamics_model.get(
                "batch_size", self.cfg.overrides.model_batch_size
            )
        else:
            print(use_untrained_model)
            self.dynamics_model = create_one_dim_tr_model_overriden(
                self.cfg,
                self.env,
                self.env.observation_space.shape,
                self.env.action_space.shape,
                model_dir=self.model_path if not use_untrained_model else None,
            )
        self.model_env = mbrl.models.ModelEnv(
            self.env,
            self.dynamics_model,
            term_fn,
            reward_fn,
            generator=torch.Generator(self.dynamics_model.device),
        )

        rng = np.random.default_rng(seed=self.cfg.seed)
        torch_generator = torch.Generator(device=self.cfg.device)
        if self.cfg.seed is not None:
            torch_generator.manual_seed(self.cfg.seed)
        use_double_dtype = self.cfg.algorithm.get("normalize_double_precision", False)
        dtype = np.double if use_double_dtype else np.float32
        self.replay_buffer = create_overriden_replay_buffer(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            rng=rng,
            obs_type=dtype,
            action_type=dtype,
            reward_type=dtype,
            load_dir=self.model_path,
        )

        self.dataset_train, _ = mbrl.util.common.get_basic_buffer_iterators(
            self.replay_buffer,
            self.replay_buffer.num_stored,
            val_ratio=0,
            ensemble_size=len(self.dynamics_model),
            shuffle_each_epoch=True,
            bootstrap_permutes=self.cfg.get("bootstrap_permutes", False),
        )

        if isinstance(self.dynamics_model.model, MultiOutputGP):
            for batch in self.dataset_train:
                self.dynamics_model.loss(batch)

        # Instanciate the agent
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
                # agent_cfg.algorithm.agent.planning_horizon = lookahead
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

        # Rendering for bikes_dynamics
        if isinstance(self.env.unwrapped, Bikes):
            self.viewer = None
            screen_ydim = 650
            screen_xdim = int(
                screen_ydim
                * abs(
                    (
                        self.env.get_wrapper_attr("longitudes")[1]
                        - self.env.get_wrapper_attr("longitudes")[0]
                    )
                    / (
                        self.env.get_wrapper_attr("latitudes")[0]
                        - self.env.get_wrapper_attr("latitudes")[1]
                    )
                )
            )
            self.screen_dim = (screen_xdim, screen_ydim)
            self.scale = np.abs(
                self.screen_dim
                / np.array(
                    [
                        self.env.get_wrapper_attr("longitudes")[1]
                        - self.env.get_wrapper_attr("longitudes")[0],
                        self.env.get_wrapper_attr("latitudes")[0]
                        - self.env.get_wrapper_attr("latitudes")[1],
                    ]
                )
            )
            self.offset = np.array(
                [
                    self.env.get_wrapper_attr("longitudes")[0],
                    self.env.get_wrapper_attr("latitudes")[0],
                ]
            )
            self.screen = None
            self.clock = None
            self.isopen = True

    def test_agent(self):
        """
        Launch some trials of the loaded agent over a given environment.
        Useful to visualize the agent performance
        You can also compare with a random agent
        """

        # create env and random seed
        observation, info = self.env.reset(seed=42)

        all_n_steps = []
        all_rewards = []

        n_steps = 0
        rewards = 0
        for env_step in range(300):
            print(
                f"Total env step: {env_step} Episode: {len(all_n_steps)} Current episode step: {n_steps}"
            )
            action = self.agent.act(
                observation
            )  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = self.env.step(action)
            print(f"Step Reward: {reward}")

            n_steps += 1
            rewards += reward

            if terminated or truncated:
                all_n_steps.append(n_steps)
                all_rewards.append(rewards)
                sleep(0.5)
                observation, info = self.env.reset()
                n_steps = 0
                rewards = 0
                print(f"Episode Reward: {rewards}")
        self.env.close()

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

    def test_model_vs_env(self):
        """
        Visualize the different trajectories taken by the agent in the real env vs in the model env
        TODO: Make sure if it works for general env (should work for the handmade ones at least)
        """

        # create env and random seed
        observation, info = self.env.reset(seed=42)

        real_trajectories = []
        model_trajectories = []

        env_step = 0
        terminated = False
        truncated = False
        while not truncated and not terminated:
            env_step += 1
            print("--------------------- \n" f"Env step: {env_step}")

            # Env step
            from time import time

            start = time()
            action = self.agent.act(observation)
            model_observation = {
                "obs": np.expand_dims(observation.copy(), axis=0),
                "propagation_indices": None,
            }
            model_action = np.expand_dims(action.copy(), axis=0)
            print("Act time: ", time() - start)
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            real_trajectories.append([observation, action, new_obs])
            print("Real env")
            print(
                f"Obs: {observation} \n"
                f"Action: {action}\n"
                f"New_obs: {new_obs}\n"
                f"Reward: {reward}"
            )
            if hasattr(self.env.unwrapped, "delta_bikes"):
                print(
                    f"Old bikes distr: {observation[self.env.get_wrapper_attr('map_obs')['bikes_distr']]}"
                    f"Delta_bikes: {self.env.get_wrapper_attr('delta_bikes')} \n"
                    f"Pre-new_bikes distr: {self.env.get_wrapper_attr('previous_bikes_distr')} \n"
                    f"New bikes distr: {new_obs[self.env.get_wrapper_attr('map_obs')['bikes_distr']]}"
                )

            observation = new_obs

            model_observation_ = {
                "obs": model_observation["obs"].copy(),
                "propagation_indices": None,
            }
            model_action_ = model_action.copy()
            next_observs, rewards, dones, new_model_observation = self.model_env.step(
                model_action_, model_observation_
            )
            model_observation = model_observation["obs"][0]
            model_observation_ = model_observation_["obs"][0].detach().numpy()
            new_model_observation = new_model_observation["obs"][0]
            next_observs = next_observs.detach().numpy()[0]
            rewards = rewards.detach().numpy()
            model_trajectories.append(
                [model_observation, model_action, new_model_observation]
            )
            print("Model env")
            print(
                f"Obs: {model_observation} \n"
                f"Action: {model_action} \n"
                f"New_obs: {next_observs}\n"
                f"Rewards: {rewards}"
            )
            if hasattr(self.env.unwrapped, "delta_bikes"):
                print(
                    f"Old bikes distr: {model_observation[self.env.get_wrapper_attr('map_obs')['bikes_distr']]}"
                    f"Delta_bikes: {model_observation_[self.env.get_wrapper_attr('map_obs')['bikes_distr']] - model_observation[self.env.get_wrapper_attr('map_obs')['bikes_distr']]} \n"
                    f"Pre-new_bikes distr: {model_observation_[self.env.get_wrapper_attr('map_obs')['bikes_distr']]} \n"
                    f"New bikes distr: {next_observs[self.env.get_wrapper_attr('map_obs')['bikes_distr']]}"
                )

        env_states = [traj[0] for traj in real_trajectories]
        env_states.append(real_trajectories[-1][-1])
        env_states = np.array(env_states)
        model_states = [traj[0] for traj in model_trajectories]
        model_states.append(model_trajectories[-1][-1])
        model_states = np.array(model_states)

        # Plot 2D trajectories
        if len(observation) == 2:
            # Plot env states:
            env_x = env_states[:, 0]
            env_y = env_states[:, 1]
            plt.scatter(env_x, env_y)
            plt.plot(env_x, env_y, label="Real env trajectory")
            plt.scatter(env_x[0], env_y[0], c="red", label="starting state")
            for i in range(len(env_x)):
                plt.annotate(f"s{i}", (env_x[i], env_y[i]))

            # Plot model states:
            model_x = model_states[:, 0]
            model_y = model_states[:, 1]
            plt.scatter(model_x, model_y)
            plt.plot(model_x, model_y, label="Model env trajectory")
            plt.scatter(model_x[0], model_y[0], c="red")
            for i in range(len(model_x)):
                plt.annotate(f"s{i}", (model_x[i], model_y[i]))

            # Winning state:
            state_x = np.concatenate([env_x, model_x])
            state_y = np.concatenate([env_y, model_y])
            for x, y in zip(state_x, state_y):
                if self.env.is_winning_state([x, y]):
                    plt.scatter(x, y, c="green", label="winning state")

            low = self.env.observation_space.low
            high = self.env.observation_space.high
            plt.xlim(low[0], high[0])
            plt.ylim(low[1], high[1])
            plt.xlabel("x")
            plt.ylabel("y")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()
            self.env.close()

        # If trajectories are of a higher dimension
        else:
            # Define a vector of reference
            ref_vector = np.ones(self.env.observation_space.shape[0])

            # Plot env states:
            env_rhos = [np.linalg.norm(state) for state in env_states]
            env_angles = [angle_between(state, ref_vector) for state in env_states]
            plt.scatter(env_rhos, env_angles)
            plt.plot(env_rhos, env_angles, label="Real env trajectory")
            plt.scatter(env_rhos[0], env_angles[0], c="red", label="starting state")
            for i in range(len(env_rhos)):
                plt.annotate(f"s{i}", (env_rhos[i], env_angles[i]))

            # Plot model states:
            model_rhos = [np.linalg.norm(state) for state in model_states]
            model_angles = [angle_between(state, ref_vector) for state in model_states]
            plt.scatter(model_rhos, model_angles)
            plt.plot(model_rhos, model_angles, label="Model env trajectory")
            plt.scatter(model_rhos[0], model_angles[0], c="red")
            for i in range(len(model_rhos)):
                plt.annotate(f"s{i}", (model_rhos[i], model_angles[i]))

            # Winning state:
            all_states = np.concatenate([env_states, model_states])
            for state in all_states:
                if self.env.is_winning_state(state):
                    rho = np.linalg.norm(state)
                    angle = angle_between(state, ref_vector)
                    plt.scatter(rho, angle, c="green", label="winning state")

            plt.xlim(0, self.env.observation_space.high[0] * 2)
            plt.ylim(0, 2 * np.pi)
            plt.xlabel("norm of the vector state")
            plt.ylabel("Angle of the vector state (1-vector as reference)")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.show()
            self.env.close()

    def test_model_sparsity(self):
        """
        Gives an idea of how much each new_state dimension is correlated to
        the previous state and action dimensions in the learned model_env.
        A good model should typically show the same Bayesian Dependencies structure than
        the actual environment.
        The "correlation values" are computed by changing a single dimension of the
        state-action space multiple times and see how it changes the given next state output.
        Repeat for each dimensions to obtain 2 correlation matrices of the form:
        state space x next_state space and action space x next_state space.
        Each row of the matrix is normalized by its diagonal entry. It makes sense for the
        environments we are dealing for now (continuous grid-like envs). But might need to
        be changed in the future.

        Careful, this function only gives an idea of the real correlation between the
        state_action space and the next_state space.

        TODO: Make sure this function works for a general model and env
        For now, should be running for any model and env with
        action and state spaces like gym.spaces.Box like.
        """

        obs_dimension = self.env.observation_space.shape[0]
        action_dimension = self.env.action_space.shape[0]
        initial_state = self.env.observation_space.sample()
        initial_action = np.expand_dims(self.env.action_space.sample(), axis=0)

        # Dependencies of each state dimension over the action space
        actions_dependencies = {}
        for i in range(action_dimension):
            action_ref = self.env.action_space.sample()
            action_ref = np.expand_dims(action_ref, axis=0)

            print(f"Testing actions dependency of next state {i}")

            # Model env step
            model_obs = {
                "obs": np.expand_dims(initial_state, axis=0),
                "propagation_indices": None,
            }
            _, _, _, model_obs = self.model_env.step(action_ref, model_obs)

            next_states_ref = model_obs["obs"][0].detach().numpy().copy()
            next_state_ref = next_states_ref[i]

            deltas = []
            for j in range(action_dimension):
                action = action_ref.copy()
                low = self.env.action_space.low[j]
                high = self.env.action_space.high[j]

                delta = 0
                for k in range(100):  # Arbitrary
                    action[:, j] = low + (high - low) / (k + 1)
                    model_obs = {
                        "obs": np.expand_dims(initial_state, axis=0),
                        "propagation_indices": None,
                    }
                    _, _, _, model_obs = self.model_env.step(action, model_obs)
                    next_states = model_obs["obs"][0].detach().numpy().copy()
                    next_state = next_states[i]

                    delta += abs(next_state - next_state_ref)
                deltas.append(delta)

            norm = deltas[i]
            deltas = list(map(lambda x: round(x / norm, 3), deltas))
            actions_dependencies[f"state_{i}_action_dependencies"] = deltas

        # Dependencies of each state dimension over the state space
        states_dependencies = {}
        for i in range(obs_dimension):
            state_ref = self.env.observation_space.sample()
            print(f"Testing states dependency of next state {i}")

            # Model env step
            model_obs = {
                "obs": np.expand_dims(state_ref, axis=0),
                "propagation_indices": None,
            }
            _, _, _, model_obs = self.model_env.step(initial_action, model_obs)

            next_states_ref = model_obs["obs"][0].detach().numpy().copy()
            next_state_ref = next_states_ref[i]

            deltas = []
            for j in range(action_dimension):
                state = state_ref.copy()
                low = self.env.observation_space.low[j]
                high = self.env.observation_space.high[j]

                delta = 0
                for k in range(100):  # Arbitrary
                    state[j] = low + (high - low) / (k + 1)
                    model_obs = {
                        "obs": np.expand_dims(state, axis=0),
                        "propagation_indices": None,
                    }
                    _, _, _, model_obs = self.model_env.step(initial_action, model_obs)
                    next_states = model_obs["obs"][0].detach().numpy().copy()
                    next_state = next_states[i]

                    delta += abs(next_state - next_state_ref)
                deltas.append(delta)

            norm = deltas[i]
            deltas = list(map(lambda x: round(x / norm, 3), deltas))
            states_dependencies[f"state_{i}_state_dependencies"] = deltas

        for key, value in actions_dependencies.items():
            print(key, value)
        for key, value in states_dependencies.items():
            print(key, value)

    def model_weights_dependencies(self, verbose=True):
        """
        This function gives another idea of the correlation between the
        next_state space and the state_action space in the learned model_env.
        For each nex_state output, we want to observe the total weight contribution
        of each state_action input.

        Careful, for the moment this function only looks at the weights (not the bias,
        residual layers or any fancy NN architecture), therefore it's hard to tell whether
        or not this function gives a relevant result...
        TODO: Make this function more reliable
        """

        all_weights = get_weights_model(self.model_env.dynamics_model.model)

        if verbose:
            for key, values in all_weights.items():
                if key == "hidden_weights" or key == "hidden_biases":
                    for i, layer in enumerate(values):
                        print(f"{key}_{i}: {layer.shape}")
                else:
                    print(key, values.shape)

        deterministic = True
        is_skip = False
        if "mean_and_logvar_weights" in all_weights.keys():
            if "min_logvar" in all_weights.keys():
                deterministic = False
            final_weights = all_weights["mean_and_logvar_weights"]
            start_weights = all_weights["hidden_weights"][0]
            weights = all_weights["hidden_weights"][1:]
        else:
            final_weights = all_weights["hidden_weights"][-1]
            start_weights = all_weights["hidden_weights"][0]
            weights = all_weights["hidden_weights"][1:-1]

        if "skip" in all_weights.keys():
            is_skip = True  # Unused for now
            start_weights = start_weights.T
            for i, weight in enumerate(weights):
                weights[i] = weight.T

        for next_state in range(self.env.observation_space.shape[0]):
            print("---------------------------------------")
            print(f"Weights associated to next state {next_state}")

            # Nope
            if not deterministic:
                final_mean = final_weights[:, next_state]
                final_logvar = final_weights[
                    :, self.env.observation_space.shape[0] + next_state
                ]

                for weight in weights:
                    final_mean = np.matmul(final_mean, weight)
                    final_logvar = np.matmul(final_logvar, weight)

                associated_weights_mean = []
                associated_weights_logvar = []

                for weight in start_weights:
                    associated_weights_mean.append(np.matmul(final_mean, weight))
                    associated_weights_logvar.append(np.matmul(final_logvar, weight))

                if is_skip:
                    pass

                # Look only relevant weights
                mean_mean = np.mean(np.abs(associated_weights_mean))
                max_mean = np.argpartition(np.abs(associated_weights_mean), -2)[-2:]
                mean_logvar = np.mean(np.abs(associated_weights_logvar))
                max_logvar = np.argpartition(np.abs(associated_weights_logvar), -2)[-2:]

                print(
                    f"Relevant action state inputs for mean next state {next_state}: {[i for i, param in enumerate(associated_weights_mean) if abs(param)>mean_mean]}, Max: {max_mean}"
                )
                print(
                    f"Relevant action state inputs for logvar next state {next_state}: {[i for i, param in enumerate(associated_weights_logvar) if abs(param)>mean_logvar]}, Max: {max_logvar}"
                )

            else:
                final_weight = final_weights[next_state]

                for weight in weights:
                    final_weight = np.matmul(final_weight, weight)

                associated_weights = []
                for weight in start_weights:
                    associated_weights.append(np.matmul(final_weight, weight))

                if is_skip:
                    pass

                # Look only relevant weights
                next_mean = np.mean(np.abs(associated_weights))
                next_max = np.argpartition(np.abs(associated_weights), -2)[-2:]

                print(
                    f"Relevant action state inputs for next state {next_state}: {[i for i, param in enumerate(associated_weights) if abs(param)>next_mean]}, Max: {next_max}"
                )

    def test_model(self):
        base_env = self.env.unwrapped
        assert isinstance(base_env, ContinuousHyperGrid)

        dataset_train, dataset_val = mbrl.util.common.get_basic_buffer_iterators(
            self.replay_buffer,
            self.replay_buffer.num_stored,
            val_ratio=0,
            ensemble_size=len(self.dynamics_model),
            shuffle_each_epoch=True,
            bootstrap_permutes=self.cfg.get("bootstrap_permutes", False),
        )

        for batch in dataset_train:
            obs, act, next_obs, rewards, _, _ = batch.astuple()
            train_x = torch.cat(
                [
                    torch.tensor(obs, dtype=torch.float32),
                    torch.tensor(act, dtype=torch.float32),
                ],
                axis=-1,
            )
            target = torch.tensor(next_obs, dtype=torch.float32)

        with torch.no_grad():
            self.dynamics_model.eval()
            x_dim = self.dynamics_model.model.in_size
            # a = np.ones(x_dim)*self.env.observation_space.low[0]
            # b = np.ones(x_dim)*self.env.observation_space.high[0]
            # test_x = torch.tensor(np.linspace(a, b, 100), dtype=torch.float32)
            test_x = train_x
            observed_preds = self.dynamics_model.forward(test_x)

        import matplotlib

        matplotlib.use("TkAgg")
        f, ax = plt.subplots(1, x_dim, figsize=(4, 3))
        for i in range(x_dim):
            # Plot training data as black stars
            j = i % (x_dim // 2)
            ax[i].plot(train_x[:, i].numpy(), target[:, j].numpy(), "k*")
            if isinstance(self.dynamics_model.model, MultiOutputGP):
                ax[i].plot(
                    test_x[:, i].numpy(), observed_preds[j].mean.detach().numpy(), "bo"
                )
                lower, upper = observed_preds[j].confidence_region()
                ax[i].fill_between(
                    test_x[:, i].numpy(),
                    lower.detach().numpy(),
                    upper.detach().numpy(),
                    alpha=0.5,
                )
                ax[i].legend(["Observed Data", "Mean", "Confidence"])
            else:
                ax[i].plot(test_x[:, i].numpy(), observed_preds[:, j].numpy(), "b")
        plt.show()

    def render_bikes(
        self, model_next_distr, true_reward=None, model_reward=None, pre_obs=False, total_failed_bikes=None, n_failed_bikes=None
    ):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.screen_dim)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(self.screen_dim)
        self.surf.fill(BLACK)

        city_map = pygame.image.load("src/env/bikes_data/louisville_map.png")
        city_size = city_map.get_size()
        city_real_dim = [-85.9, -85.55, 38.15, 38.35]
        city_scale = np.abs(
            city_size
            / np.array(
                [
                    city_real_dim[1] - city_real_dim[0],
                    city_real_dim[3] - city_real_dim[2],
                ]
            )
        )
        city_offset = np.array([city_real_dim[0], city_real_dim[2]])
        x = (self.env.get_wrapper_attr("longitudes") - city_offset[0]) * city_scale[0]
        # Warning: pygame has a reversed y axis
        y = (
            city_size[1]
            - (self.env.get_wrapper_attr("latitudes") - city_offset[1]) * city_scale[1]
        )
        cropped_region = (x[0], y[1], x[1] - x[0], y[0] - y[1])
        city_map = city_map.subsurface(cropped_region)
        city_map = pygame.transform.scale(city_map, self.screen_dim)
        self.surf.blit(city_map, (0, 0))

        # Added bikes from depot:
        if pre_obs:
            font_size = 10
            font = pygame.font.SysFont("Arial", font_size)
            depot_coord = (self.screen_dim[0] - 50, self.screen_dim[1] - 50)
            if self.env.get_wrapper_attr("delta_bikes") is not None:
                for i, added_bikes in enumerate(
                    self.env.get_wrapper_attr("delta_bikes")
                ):
                    if added_bikes > 0:
                        coord = self.env.get_wrapper_attr("centroid_coords")[i]
                        coord = (coord[1], coord[0])
                        new_coord = (coord - self.offset) * self.scale
                        new_coord[1] = self.screen_dim[1] - new_coord[1]
                        width = 1  # added_bikes
                        self.env.unwrapped._draw_arrow(
                            self.surf,
                            pygame.Vector2(depot_coord[0], depot_coord[1]),
                            pygame.Vector2(new_coord[0], new_coord[1]),
                            PRETTY_RED,
                            width,
                            2 + min(5 * width, 10 + width),
                        )
                        txtsurf = font.render(str(added_bikes), True, DARK_RED)
                        alpha = uniform(0.25, 0.5)
                        text_coord = depot_coord + alpha * (new_coord - depot_coord)
                        self.surf.blit(
                            txtsurf,
                            (
                                text_coord[0] - font_size / 3.5,
                                text_coord[1] - font_size / 1.5,
                            ),
                        )
            pygame.draw.circle(self.surf, PRETTY_RED, depot_coord, 10)
            txtsurf = font.render("DEPOT", True, BLACK)
            self.surf.blit(
                txtsurf,
                (depot_coord[0] - font_size / 1.5, depot_coord[1] - font_size / 1.5),
            )
            real_next_distr = self.env.get_wrapper_attr("previous_bikes_distr")
        else:
            real_next_distr = self.env.get_wrapper_attr("state")["bikes_distr"]

        # Predicted bikes_distr vs real bikes_distr
        font_size = 15
        font = pygame.font.SysFont("Arial", font_size)
        model_next_distr = np.round(model_next_distr).astype(int)
        for coord, real_bikes, model_bikes in zip(
            self.env.get_wrapper_attr("centroid_coords"),
            real_next_distr,
            model_next_distr,
        ):
            coord = (coord[1], coord[0])
            new_coord = (coord - self.offset) * self.scale
            new_coord[1] = self.screen_dim[1] - new_coord[1]
            radius = 15
            color = PRETTY_GREEN if real_bikes == model_bikes else RED
            pygame.draw.circle(self.surf, color, new_coord, radius)
            txtsurf = font.render(f"{real_bikes}-{model_bikes}", True, BLACK)
            self.surf.blit(
                txtsurf,
                (new_coord[0] - font_size / 3.5, new_coord[1] - font_size / 1.5),
            )

        # Legend:
        font_size = 15
        font = pygame.font.SysFont("Arial", font_size)
        shift = self.env.unwrapped.get_timeshift()
        title_str = (
            (
                f"Shift {shift[0]}:{shift[1]} Day: {int(self.env.get_wrapper_attr('state')['day'])} "
                f"({int(self.env.get_wrapper_attr('state')['day_of_week'])}/7) "
                f"Month: {int(self.env.get_wrapper_attr('state')['month'])}"
            )
            if shift is not None
            else (
                f"Shift: {shift} Day: {int(self.env.get_wrapper_attr('state')['day'])} "
                f"({int(self.env.get_wrapper_attr('state')['day_of_week'])}/7) "
                f"Month: {int(self.env.get_wrapper_attr('state')['month'])}"
            )
        )
        title = font.render(title_str, True, BLACK)
        self.surf.blit(title, (self.screen_dim[0] // 2, 0))

        if true_reward and model_reward:
            font_size = 15
            font = pygame.font.SysFont("Arial", font_size)
            text = font.render(
                f"True reward: {true_reward} Model reward: {model_reward}", True, BLACK
            )
            self.surf.blit(text, (10 - font_size / 3.5, 20 - font_size / 1.5))


        if total_failed_bikes and n_failed_bikes:
            text = font.render(
                f"Total misspred bikes: {total_failed_bikes}(+{n_failed_bikes} this round)", True, BLACK
            )
            self.surf.blit(text, (10 - font_size / 3.5, 40 - font_size / 1.5))


        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()

    def check_preprocessed_bikes(self):
        observation, info = self.env.reset(seed=42)

        episode_step = 0
        for env_step in range(self.num_steps):
            print(f"Total env step: {env_step} Episode step: {episode_step}")

            # Take action
            action = self.env.get_wrapper_attr('action_space').sample()
            model_observation = {
                "obs": np.expand_dims(observation.copy(), axis=0),
                "propagation_indices": None,
            }
            model_action = np.expand_dims(action.copy(), axis=0)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            model_in, preprocessed_obs = self.dynamics_model._get_model_input(model_observation["obs"].copy(), model_action.copy())
            preprocessed_obs = preprocessed_obs.clone().detach().numpy()[0].astype(int)
            pre_bikes_distr = preprocessed_obs[self.env.get_wrapper_attr('map_obs')['bikes_distr']]
            assert np.all(pre_bikes_distr == self.env.get_wrapper_attr("previous_bikes_distr"))
            (
                next_model_observs,
                model_rewards,
                model_dones,
                next_model_state,
            ) = self.model_env.step(model_action, model_observation)
            next_model_obs = next_model_state["obs"][0].detach().numpy()
            next_model_observs = next_model_observs[0] #.detach().numpy()[0]
            model_reward = model_rewards[0,0] #.detach().numpy()[0, 0]
            model_done = model_dones[0,0] #.detach().numpy()[0, 0]

            assert model_done == terminated
            assert np.all(next_model_obs == next_model_observs)

            observation = next_obs
            episode_step += 1

            if terminated or truncated:
                observation, info = self.env.reset()
                episode_step = 0
        self.env.close()


    def test_bikes_learned_dynamics(self, plan_whole_episode=True):
        """Compare the predicted dynamics of model env with the real env"""

        observation, info = self.env.reset(seed=42)
        model_obs = observation

        episode_step = 0
        total_failed_bikes = 0
        for env_step in range(self.num_steps):
            print(f"Total env step: {env_step} Episode step: {episode_step}")

            # Take action
            model_obs = model_obs if plan_whole_episode else observation.copy()
            model_observation = {
                "obs": np.expand_dims(model_obs, axis=0),
                "propagation_indices": None,
            }
            action = self.agent.act(model_obs)
            model_action = np.expand_dims(action.copy(), axis=0)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.render_bikes(
                model_obs[self.env.get_wrapper_attr("map_obs")["bikes_distr"]]
                + self.env.get_wrapper_attr("delta_bikes"),
                pre_obs=True,
            )
            #sleep(2)

            (
                next_model_observs,
                model_rewards,
                model_dones,
                next_model_state,
            ) = self.model_env.step(model_action, model_observation)
            next_model_obs = next_model_state["obs"][0].detach().numpy()
            next_model_observs = next_model_observs.detach().numpy()[0]
            model_reward = model_rewards.detach().numpy()[0, 0]
            model_done = model_dones.detach().numpy()[0, 0]

            assert np.all(next_model_obs == next_model_observs)

            reward = round(reward, 2)
            model_reward = round(float(model_reward),2)
            print(
                "Real env vs Model env"
                f"real next_obs - model next_obs: {np.round(next_obs - next_model_obs, 2)} \n"
                f"True reward: {reward} vs Model reward: {model_reward} \n"
                f"True done: {terminated} vs model done: {model_done}"
            )

            next_model_distr = np.round(
                    next_model_obs[self.env.get_wrapper_attr("map_obs")["bikes_distr"]]
                )
            next_distr = self.env.get_wrapper_attr('state')['bikes_distr']
            n_failed_bikes = np.sum(np.abs(next_model_distr-next_distr))
            total_failed_bikes += n_failed_bikes

            self.render_bikes(
                next_model_distr,
                reward,
                model_reward,
                n_failed_bikes=n_failed_bikes,
                total_failed_bikes=total_failed_bikes
            )
            #input()
            #sleep(2)

            observation = next_obs
            model_obs = next_model_obs
            episode_step += 1

            if terminated or truncated:
                sleep(0.5)
                observation, info = self.env.reset()
                model_obs = observation
                episode_step = 0
                total_failed_bikes = 0
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=None,
        help="The directory where the original experiment was run.",
    )
    parser.add_argument("--lookahead", type=int, default=25)
    parser.add_argument(
        "--random_agent", default=False, action=argparse.BooleanOptionalAction
    )
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
    parser.add_argument(
        "--function",
        type=str,
        default="test_agent",
        help="Name of the function you want to use",
    )
    parser.add_argument(
        "--render",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add it if you want to render the environment",
    )
    parser.add_argument(
        "--use_phm",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add it if you want to use a 'perfect' model to solve the hypergrid env",
    )
    parser.add_argument(
        "--use_untrained_model",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add it if you want to use a not-trained model",
    )
    args = parser.parse_args()

    visualizer = AdaptedVisualizer(
        lookahead=args.lookahead,
        results_dir=args.experiments_dir,
        random_agent=args.random_agent,
        num_steps=args.num_steps,
        num_model_samples=args.num_model_samples,
        model_subdir=args.model_subdir,
        render=args.render,
        use_perfect_hypergrid_model=args.use_phm,
        use_untrained_model=args.use_untrained_model,
    )

    if args.function == "test_agent":
        visualizer.test_agent()
    elif args.function == "test_model":
        visualizer.test_model()
    elif args.function == "test_model_vs_env":
        visualizer.test_model_vs_env()
    elif args.function == "test_model_sparsity":
        visualizer.test_model_sparsity()
    elif args.function == "model_weights_dependencies":
        visualizer.model_weights_dependencies()
    elif args.function == "test_bikes_learned_dynamics":
        visualizer.test_bikes_learned_dynamics()
    elif args.function == "check_preprocessed_bikes":
        visualizer.check_preprocessed_bikes()
    else:
        raise ValueError("There is no such function implemented by the Visualizer")
