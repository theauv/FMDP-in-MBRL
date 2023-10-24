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
from src.env.constants import BLACK, GREEN, RED, WHITE
from src.util.util import create_one_dim_tr_model_overriden, get_weights_model


VisData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
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

        self.dynamics_model = create_one_dim_tr_model_overriden(
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
        model_observation = {
            "obs": np.expand_dims(observation, axis=0),
            "propagation_indices": None,
        }

        real_trajectories = []
        model_trajectories = []

        env_step = 0
        terminated = False
        truncated = False
        while not truncated and not terminated:
            env_step += 1
            print("--------------------- \n" f"Env step: {env_step}")

            # Env step
            action = self.agent.act(observation)
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            real_trajectories.append([observation, action, new_obs])
            print("Real env")
            print(f"Obs: {observation}, Action: {action}, New_obs: {new_obs}")
            observation = new_obs

            # Model_env step
            if env_step == 1 or np.all(model_observation["obs"][0] == observation):
                model_action = action
            else:
                model_action = self.agent.act(model_observation["obs"][0])
            model_action = np.expand_dims(model_action, axis=0)

            old_model_observation = model_observation["obs"][0].copy()

            _, _, _, model_observation = self.model_env.step(
                model_action, model_observation
            )
            model_observation["obs"] = model_observation["obs"].detach().numpy()
            model_trajectories.append(
                [old_model_observation, model_action[0], model_observation["obs"][0]]
            )
            print("Model env")
            print(
                f"Obs: {old_model_observation}, Action: {model_action}, New_obs: {model_observation['obs'][0]}"
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

        Careful, for the moment thi function only looks at the weights (not the bias,
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
    args = parser.parse_args()

    visualizer = AdaptedVisualizer(
        lookahead=args.lookahead,
        results_dir=args.experiments_dir,
        random_agent=args.random_agent,
        num_steps=args.num_steps,
        num_model_samples=args.num_model_samples,
        model_subdir=args.model_subdir,
        render=args.render,
    )

    if args.function == "test_agent":
        visualizer.test_agent()
    elif args.function == "test_model_vs_env":
        visualizer.test_model_vs_env()
    elif args.function == "test_model_sparsity":
        visualizer.test_model_sparsity()
    elif args.function == "model_weights_dependencies":
        visualizer.model_weights_dependencies()
    else:
        raise ValueError("There is no such function implemented by the Visualizer")
