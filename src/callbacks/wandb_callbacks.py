from typing import Union
import wandb

from mbrl.models import Model

# TODO: Make sure this is not memory-vore to update and plot same table over time !!!!


class CallbackWandb:
    """
    Class dealing with the generated callbacks during a training run

    Attributes
    ----------
    Each attribute is a table that gather a given data that has to be sent to 
    the API (epoch, loss, reward, ...)

    """
    def __init__(self) -> None:
        self.epoch_data = []
        self.total_avg_loss_data = []
        self.eval_score_data = []
        self.best_val_score_data = []
        self.episode_steps_data = []
        self.episode_reward_data = []

    def delete_alias_artifacts(self, run_id):
        # TODO!!
        pass

    def model_train_callback(
        self, model: Model, train_iter: int, epoch: int, total_avg_loss: float, eval_score: float, best_val_score: float
    ):
        """
        update associated tables and add them to the plots in wandb
        """

        self.epoch_data.append([train_iter, epoch])
        self.total_avg_loss_data.append([train_iter, total_avg_loss])
        self.eval_score_data.append([train_iter, eval_score])
        self.best_val_score_data.append([train_iter, best_val_score])

        datas = {
            "epoch": self.epoch_data,
            "total_avg_loss": self.total_avg_loss_data,
            "eval_score": self.eval_score_data,
            "best_eval": self.best_val_score_data,
        }

        plots = {}
        for name, data in datas.items():
            table = wandb.Table(data=data, columns=["train_iteration", name])
            title = f"Model_{name}"
            plots[title] = wandb.plot.line(table, "train_iteration", name, title=title)
        wandb.log(plots)

    def agent_callback(self, episode: int, episode_steps: int, episode_reward: float):
        """
        update associated tables and add them to the plots in wandb
        """

        self.episode_steps_data.append([episode, episode_steps])
        self.episode_reward_data.append([episode, episode_reward])

        datas = {
            "episode_steps": self.episode_steps_data,
            "episode_reward": self.episode_reward_data,
        }

        plots = {}
        for name, data in datas.items():
            table = wandb.Table(data=data, columns=["episode", name])
            title = f"Agent_{name}"
            plots[title] = wandb.plot.line(table, "episode", name, title=title)
        wandb.log(plots)
