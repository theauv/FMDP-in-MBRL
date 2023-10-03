import math

import numpy as np
import torch

from env.hypergrid import GRID_DIM, GRID_SIZE, SIZE_END_BOX


def maze(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    game_limit = 10.0

    x, y = next_obs[:, 0], next_obs[:, 1]

    not_done = (
        (x > -game_limit) * (x < game_limit) * (y > -game_limit) * (y < game_limit)
    )

    done = ~not_done
    done = done[:, None]  # augment dimension
    return done


def hypergrid(action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    done = torch.ones(next_obs.shape[0], dtype=bool)
    for i in range(GRID_DIM):
        done *= (next_obs[:, i] > (GRID_SIZE - SIZE_END_BOX))

    done = done[:, None]  # augment dimension
    return done
