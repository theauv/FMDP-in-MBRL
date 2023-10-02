import torch

from . import termination_fns


def maze(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(action.shape) == 2

    penalty_step = -0.1

    return (~termination_fns.maze(action, next_obs) * penalty_step).float().view(-1, 1)
