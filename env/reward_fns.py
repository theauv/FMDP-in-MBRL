import torch

from . import termination_fns
from env.hypergrid import STEP_PENALTY


def maze(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(action.shape) == 2

    penalty_step = -0.1

    return (~termination_fns.maze(action, next_obs) * penalty_step).float().view(-1, 1)


def hypergrid(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(action.shape) == 2

    return (
        (~termination_fns.hypergrid(action, next_obs) * STEP_PENALTY)
        .float()
        .view(-1, 1)
    )
