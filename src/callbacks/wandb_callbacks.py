from typing import Dict

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
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
    ) -> None:
        """
        Define the different metrics usseful to track and plot

        :param with_tracking: if we don't want to track and plot in wandb, default is True
        """

        self.max_traj_iterations = max_traj_iterations
        self.with_tracking = with_tracking
        self.env_step = 0
        self.model_out_size = model_out_size

        # Define new metrics for each tracked values
        if self.with_tracking:
            wandb.define_metric("env_episode", hidden=True)
            wandb.define_metric("episode_steps", step_metric="env_episode")
            wandb.define_metric("episode_reward", step_metric="env_episode")

            wandb.define_metric("train_iteration", hidden=True)
            wandb.define_metric("total_avg_loss", step_metric="train_iteration")
            wandb.define_metric("eval_score", step_metric="train_iteration")
            wandb.define_metric("best_eval_score", step_metric="train_iteration")

            wandb.define_metric("trajectory_optimizer_iteration", hidden=True)
            wandb.define_metric(
                "trajectory_optimizer_eval",
                step_metric="trajectory_optimizer_iteration",
            )

    def model_train_callback(
        self,
        model: Model,
        train_iter: int,
        epoch: int,
        total_avg_loss: float,
        eval_score: float,
        best_eval_score: float,
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

    def model_sparsity(self, which_lassonet = None, fig_loss = None, fig_theta = None, factors=None):

        if not self.with_tracking:
            print(f"Lassonet {which_lassonet}")
            plt.show()
            return
        
        if which_lassonet is not None:

            wandb.log({f'Loss pretraining lassonet {which_lassonet}' : wandb.Image(fig_loss), 
                    f'Theta pretraining lassonet {which_lassonet}': wandb.Image(fig_theta),
                    }) 
        

        if factors is not None:

            factors = np.array(factors)
            outputs = range(len(factors))
            inputs = range(np.max(factors)+1)

            G = nx.DiGraph()

            for x in inputs:
                G.add_node(f'in_{x}', pos=(0,x), color='blue')

            for x in outputs:
                G.add_node(f'out_{x}', pos=(2,x), color='red')

            for output, factor in enumerate(factors):
                print(factor)
                for input_ in factor:
                    G.add_edge(f'in_{input_}', f'out_{output}')

            pos=nx.get_node_attributes(G,'pos')
            color=nx.get_node_attributes(G,'color')
            graph_fig = plt.figure()
            nx.draw_networkx(G, with_labels = True, pos=pos, node_color=color.values())

            wandb.log({f'Learned DBN graph' : wandb.Image(graph_fig), 
            }) 
        
