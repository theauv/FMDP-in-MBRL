from typing import Dict
import wandb

from mbrl.models import Model

# TODO: Make it more modular maybe ? (Not trivial as it would mean touch mbrl functions)


class CallbackWandb:
    """
    Class dealing with the generated callbacks during a training run
    """

    def __init__(self, with_tracking: bool = True) -> None:
        """
        Define the different metrics usseful to track and plot

        :param with_tracking: if we don't want to track and plot in wandb, default is True
        """

        self.with_tracking = with_tracking

        # Define new metrics for each tracked values
        if self.with_tracking:
            wandb.define_metric("env_episode", hidden=True)
            wandb.define_metric("episode_steps", step_metric="env_episode")
            wandb.define_metric("episode_reward", step_metric="env_episode")

            wandb.define_metric("train_iteration", hidden=True)
            wandb.define_metric("total_avg_loss", step_metric="train_iteration")
            wandb.define_metric("eval_score", step_metric="train_iteration")
            wandb.define_metric("best_eval_score", step_metric="train_iteration")

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

        wandb.log(tracked_values)
