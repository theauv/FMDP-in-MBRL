import math

import torch


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
