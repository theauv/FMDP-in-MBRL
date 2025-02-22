from typing import List, Optional

import gymnasium as gym
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pyvis.network import Network
import torch
import wandb

from mbrl.models import Model

from src.env.dict_spaces_env import DictSpacesEnv

# TODO: Make it more modular maybe ? (Not trivial as it would mean touch mbrl functions)


class CallbackWandb:
    """
    TODO: Split into diferent classes?
    Class dealing with the generated callbacks during a training run
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        with_tracking: bool = True,
        max_traj_iterations: int = 0,
        model_out_size: int = None,
        plot_local: bool = False,
        centroid_coords: Optional[List] = None,
        num_epochs_train_model: Optional[int] = None,
        epsilon: Optional[float] = 1.0e-6,
    ) -> None:
        """
        Define the different metrics usseful to track and plot
        :param with_tracking: if we don't want to track and plot in wandb, default is True
        """
        self.num_epochs_train_model = num_epochs_train_model
        self.max_traj_iterations = max_traj_iterations
        self.with_tracking = with_tracking
        self.env_step = 0
        self.model_out_size = model_out_size
        self.plot_local = plot_local
        self.centroid_coords = centroid_coords
        self.epsilon = epsilon
        self.map_obs = None
        self.map_act = None
        self.embed_act = None
        if env:
            self.embed_act = getattr(env.unwrapped, "embed1d_act", None)
            if isinstance(env.unwrapped, DictSpacesEnv):
                self.map_obs = env.unwrapped.map_obs
                self.map_act = env.unwrapped.map_act

        # Define new metrics for each tracked values
        if self.with_tracking:
            wandb.define_metric("env_episode", hidden=True)
            wandb.define_metric("episode_steps", step_metric="env_episode")
            wandb.define_metric("episode_reward", step_metric="env_episode")

            # DEBUG Bikes Benchmark only
            wandb.define_metric("env_step", hidden=True)
            wandb.define_metric("env_episode", step_metric="env_step")
            wandb.define_metric("step_reward", step_metric="env_step")
            wandb.define_metric("step_time_0_6", hidden=True)
            wandb.define_metric("step_reward_0_6", step_metric="step_time_0_6")
            wandb.define_metric("step_time_6_12", hidden=True)
            wandb.define_metric("step_reward_6_12", step_metric="step_time_6_12")
            wandb.define_metric("step_time_12_18", hidden=True)
            wandb.define_metric("step_reward_12_18", step_metric="step_time_12_18")
            wandb.define_metric("step_time_18_24", hidden=True)
            wandb.define_metric("step_reward_18_24", step_metric="step_time_18_24")

            wandb.define_metric("train_iteration", hidden=True)
            wandb.define_metric("total_avg_loss", step_metric="train_iteration")
            wandb.define_metric("eval_score", step_metric="train_iteration")
            wandb.define_metric("best_eval_score", step_metric="train_iteration")
            wandb.define_metric("train_r2_score", step_metric="train_iteration")
            wandb.define_metric("eval_r2_score", step_metric="train_iteration")

            wandb.define_metric("trajectory_optimizer_iteration", hidden=True)
            wandb.define_metric(
                "trajectory_optimizer_eval",
                step_metric="trajectory_optimizer_iteration",
            )

            wandb.define_metric("train_epoch", hidden=True)
            wandb.define_metric("total_avg_loss", step_metric="train_epoch")
            wandb.define_metric("eval_score", step_metric="train_epoch")
            wandb.define_metric("best_eval_score", step_metric="train_epoch")
            wandb.define_metric("train_r2_score", step_metric="train_epoch")
            wandb.define_metric("eval_r2_score", step_metric="train_epoch")

    def env_callback(self, env):
        if not self.with_tracking:
            if self.plot_local and env.render_mode == "rgb_array":
                matplotlib.use("Agg")
                plt.imshow(env.render("rgb_array"))
                plt.show()
            return

        image = wandb.Image(env.render("rgb_array"))
        wandb.log({"Env": image})

    def model_train_callback(
        self,
        model: Model,
        train_iter: int,
        epoch: int,
        total_avg_loss: float,
        eval_score: float,
        best_eval_score: float,
        train_r2_score: Optional[float] = None,
        eval_r2_score: Optional[float] = None,
    ):
        """
        Plot the training scores of the model
        This function is meant to be pass to the ModelTrainer as an argument
        """

        if not self.with_tracking:
            return

        tracked_values = {
            "total_avg_loss": total_avg_loss,
            "eval_score": eval_score,
            "best_eval_score": best_eval_score,
            "train_iteration": train_iter,
        }

        if train_r2_score is not None:
            tracked_values.update({"train_r2_score": train_r2_score})
        if eval_r2_score is not None:
            tracked_values.update({"eval_r2_score": eval_r2_score})

        if self.num_epochs_train_model is not None:
            tracked_values.update(
                {"train_epoch": epoch + int(train_iter * self.num_epochs_train_model)}
            )

        wandb.log(tracked_values)

    def split_model_train_callback(
        self,
        train_iter: int,
        epoch: int,
        dyn_eval_score: float,
        rew_eval_score: float,
        dyn_r2: float,
        rew_r2: float,
    ):
        """
        Plot the training scores of the model
        This function is meant to be pass to the ModelTrainer as an argument
        """

        if not self.with_tracking:
            return

        tracked_values = {
            "dynamics_eval_score": dyn_eval_score,
            "reward_eval_score": rew_eval_score,
            "train_iteration": train_iter,
        }

        if dyn_r2 is not None:
            tracked_values.update({"dynamics_r2": dyn_r2})
        if rew_r2 is not None:
            tracked_values.update({"reward_r2": rew_r2})

        wandb.log(tracked_values)

    def model_train_callback_per_epoch(
        self,
        model: Model,
        train_iter: int,
        epoch: int,
        total_avg_loss: float,
        eval_score: float,
        best_eval_score: float,
        train_r2_score: Optional[float] = None,
        eval_r2_score: Optional[float] = None,
    ):
        """
        Plot the training scores of the model
        This function is meant to be pass to the ModelTrainer as an argument
        """

        if not self.with_tracking:
            return

        tracked_values = {
            "train_epoch": epoch,
            "total_avg_loss": total_avg_loss,
            "eval_score": eval_score,
            "best_eval_score": best_eval_score,
        }

        if train_r2_score is not None:
            tracked_values.update({"train_r2_score": train_r2_score})
        if eval_r2_score is not None:
            tracked_values.update({"eval_r2_score": eval_r2_score})

        wandb.log(tracked_values)

    def agent_callback(self, episode: int, episode_steps: int, episode_reward: float):
        """
        Plot the performance of the agent in the real environment
        update associated tables and add them to the plots in wandb
        """

        if not self.with_tracking:
            return

        tracked_values = {
            "episode_steps": episode_steps,
            "episode_reward": episode_reward,
            "env_episode": episode,
            "env_step": self.env_step,
        }

        self.episodes_steps = episode_steps

        wandb.log(tracked_values)

    def track_each_step(self, step: int, step_reward: float):
        """
        DEBUGGING PURPOSE
        Plot the performance of the agent in the real environment
        update associated tables and add them to the plots in wandb
        """

        if not self.with_tracking:
            return

        tracked_values = {"env_step": step, "step_reward": step_reward}
        wandb.log(tracked_values)

        if step % 8 < 2:
            tracked_values = {"step_time_0_6": step, "step_reward_0_6": step_reward}
            wandb.log(tracked_values)
        elif step % 8 < 4:
            tracked_values = {"step_time_6_12": step, "step_reward_6_12": step_reward}
            wandb.log(tracked_values)
        elif step % 8 < 6:
            tracked_values = {"step_time_12_18": step, "step_reward_12_18": step_reward}
            wandb.log(tracked_values)
        else:
            tracked_values = {"step_time_18_24": step, "step_reward_18_24": step_reward}
            wandb.log(tracked_values)

    def trajectory_optimizer_callback(self, population, values, iterations):
        if not self.with_tracking:
            return

        best_value = torch.max(values)

        tracked_values = {
            "best_trajectory_value": best_value,
            "trajectory_optimization_iteration": self.env_step
            * self.max_traj_iterations
            + iterations,
        }

        wandb.log(tracked_values)

    def model_sparsity(
        self, which_lassonet=None, fig_loss=None, fig_theta=None, thetas=None
    ):
        if not self.with_tracking:
            matplotlib.use("TkAgg")
            print(f"Lassonet {which_lassonet}")
            plt.show()
            return

        if which_lassonet is not None:
            wandb.log(
                {
                    f"Loss pretraining lassonet {which_lassonet}": wandb.Image(
                        fig_loss
                    ),
                    f"Theta pretraining lassonet {which_lassonet}": wandb.Image(
                        fig_theta
                    ),
                }
            )

    @staticmethod
    def dbn_graph_pyvis(factors):
        """
        Create a dbn graph visualization of the model factors
        with pyvis for a general model
        """
        n_outputs = len(factors)
        n_inputs = max(map(max, factors)) + 1
        size = 600
        input_scale = size / n_inputs
        output_scale = size / n_outputs
        input_node_size = size / (4 * n_inputs)
        output_node_size = size / (4 * n_outputs)
        font_size = min(300 / n_inputs, 10)
        net = Network(f"{size}px", select_menu=True)
        net.toggle_physics(False)

        for i in range(n_inputs):
            net.add_node(
                f"i{i}",
                x=-size / 2,
                y=i * input_scale,
                color="blue",
                size=input_node_size,
            )
        for i in range(n_outputs):
            x = size / 2
            y = i * output_scale
            net.add_node(f"o{i}", x=x, y=y, color="red", size=output_node_size)
        for output, factor in enumerate(factors):
            for input_ in factor:
                net.add_edge(f"i{input_}", f"o{output}", width=0.1)
        return net

    @staticmethod
    def dbn_bikes_graph_pyvis(factors, centroid_coords):
        """
        Create a dbn graph visualization of the model factors
        with pyvis for a model that deals with Bikes environment
        """
        graph_size = 600
        net = Network(f"{graph_size}px", select_menu=True)
        net.toggle_physics(False)

        # TODO: Write this in a modulable way
        factors = factors[: len(centroid_coords)]

        for i, centroid_coord in enumerate(centroid_coords):
            centroid_coord = np.flip(centroid_coord)
            net.add_node(
                i,
                label=i,
                x=centroid_coord[0] * graph_size ** 2,
                y=(graph_size - centroid_coord[1]) * graph_size ** 2,
                color="blue",
                size=100,
            )

        for output, factor in enumerate(factors):
            for input_ in factor:
                if input_ < len(centroid_coords) and output != input_:
                    net.add_edge(int(input_), output, width=0.1)

        return net

    def model_dbn(self, factors: List, positions=None):
        """
        Warning: positions parameter is more of a debugging parameters for the
        Bikes environment
        """
        html_file = "dbn_graph.html"
        if not self.with_tracking:
            if self.plot_local:
                if self.centroid_coords is None:
                    net = self.dbn_graph_pyvis(factors)
                else:
                    net = self.dbn_bikes_graph_pyvis(factors, self.centroid_coords)
                net.show(html_file, notebook=False)
            return
        if self.centroid_coords is None:
            net = self.dbn_graph_pyvis(factors)
        else:
            net = self.dbn_bikes_graph_pyvis(factors, self.centroid_coords)
        if self.plot_local:
            net.show(html_file, notebook=False)
        else:
            net.write_html(html_file, open_browser=False)
        wandb.log({f"DBN graph": wandb.Html(open("dbn_graph.html"), inject=False)})

    def compute_dict_pred_error(
        self, next_obs, next_model_obs, output_keys, epsilon=1.0e-6
    ):
        errors = {}
        for key, value in self.map_obs.items():
            if key != "length":
                if key not in output_keys:
                    assert np.all(next_obs[..., value] == next_model_obs[value])
                else:
                    real_out = next_obs[..., value]
                    model_out = next_model_obs[..., value]
                    if (
                        key == "bike_allocations"
                    ):  # TODO: not great to have it like this but can't think of a better way
                        tot_n_bikes = next_obs[..., self.map_obs["tot_n_bikes"]]
                        error_key = f"Missclassified pred_{key} ratio"
                        errors[error_key] = (
                            np.sum(np.abs(real_out - model_out)) / (2 * tot_n_bikes)
                            if tot_n_bikes > 0
                            else np.nan
                        )
                    else:
                        error_key = f"Mean relative error pred_{key}"
                        errors[error_key] = np.mean(
                            np.abs((model_out - real_out) / (real_out + epsilon))
                        )
        return errors

    def compute_pred_error(self, next_obs, next_model_obs, epsilon=1.0e-6):
        return np.mean(np.abs((next_model_obs - next_obs) / (next_obs + epsilon)))

    def pred_dynamics_callback(
        self,
        action,
        next_obs,
        reward,
        next_model_obs,
        model_reward,
        output_keys: Optional[List[str]] = None,
    ):
        if not self.with_tracking:
            return

        if output_keys and self.map_obs:
            errors = self.compute_dict_pred_error(
                next_obs, next_model_obs, output_keys, self.epsilon
            )
        else:
            error = self.compute_pred_error(next_obs, next_model_obs, self.epsilon)
            errors = {"Mean relative error pred_obs": error}

        reward_error = np.abs((reward - model_reward) / (reward + self.epsilon))
        errors[f"Mean relative error pred_reward"] = reward_error

        if self.env_step <= 1:
            wandb.define_metric("env_step", hidden=True)
            for key in errors.keys():
                wandb.define_metric(key, step_metric="env_step")

        errors["env_step"] = self.env_step

        if self.embed_act:
            embedded_act = self.embed_act(action)
            errors["embedded_action"] = embedded_act

        wandb.log(errors)
