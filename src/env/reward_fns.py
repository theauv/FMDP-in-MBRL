import torch

import src.env.termination_fns as termination_fns
from src.env.hypergrid import STEP_PENALTY


def maze(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    Reward function associated to the maze env

    :param action: batch of actions
    :param next_obs: batch of next_obs
    :return: batch of rewards associated to each (action, next_obs)
    """
    assert len(next_obs.shape) == len(action.shape) == 2

    penalty_step = -0.1

    return (~termination_fns.maze(action, next_obs) * penalty_step).float().view(-1, 1)


def hypergrid(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    Reward function associated to the hypergrid env

    :param action: batch of actions
    :param next_obs: batch of next_obs
    :return: batch of rewards associated to each (action, next_obs)
    """
    assert len(next_obs.shape) == len(action.shape) == 2

    return (
        (~termination_fns.hypergrid(action, next_obs) * STEP_PENALTY)
        .float()
        .view(-1, 1)
    )