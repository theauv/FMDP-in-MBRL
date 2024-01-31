from typing import List, Optional

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pyvis.network import Network
import torch
import wandb

from mbrl.models import Model

# TODO: Make it more modular maybe ? (Not trivial as it would mean touch mbrl functions)


class CallbackWandb:
    """
    Class dealing with the generated callbacks during a training run
    """

    def __init__(
        self,
        with_tracking: bool = True,
        max_traj_iterations: int = 0,
        model_out_size: int = None,
        plot_local: bool = False,
        centroid_coords: Optional[List] = None,
        num_epochs_train_model: Optional[int] = None,
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

        # Define new metrics for each tracked values
        if self.with_tracking:
            wandb.define_metric("env_episode", hidden=True)
            wandb.define_metric("episode_steps", step_metric="env_episode")
            wandb.define_metric("episode_reward", step_metric="env_episode")

            # DEBUG Bikes Benchmark only
            wandb.define_metric("env_step", hidden=True)
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
            tracked_values.update(
                {
                    "train_r2_score": train_r2_score,
                }
            )
        if eval_r2_score is not None:
            tracked_values.update(
                {
                    "eval_r2_score": eval_r2_score,
                }
            )

        if self.num_epochs_train_model is not None:
            tracked_values.update(
                {
                    "train_epoch": epoch+int(train_iter*self.num_epochs_train_model),
                }
            )

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
            tracked_values.update(
                {
                    "train_r2_score": train_r2_score,
                }
            )
        if eval_r2_score is not None:
            tracked_values.update(
                {
                    "eval_r2_score": eval_r2_score,
                }
            )

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
        self, which_lassonet=None, fig_loss=None, fig_theta=None, factors=None
    ):
        if not self.with_tracking:
            print(f"Lassonet {which_lassonet}")
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
                x=centroid_coord[0] * graph_size**2,
                y=(graph_size - centroid_coord[1]) * graph_size**2,
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
