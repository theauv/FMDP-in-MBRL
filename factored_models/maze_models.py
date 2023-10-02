from typing import Dict, Optional, Tuple, Union
from math import ceil
import omegaconf
import torch

from mbrl.models.gaussian_mlp import GaussianMLP


class FactoredGaussianMLP(GaussianMLP):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):

        self.in_size = ceil(in_size / 2)
        self.out_size = ceil(out_size / 2)

        super().__init__(
            self.in_size,
            self.out_size,
            device,
            num_layers,
            ensemble_size,
            hid_size,
            deterministic,
            propagation_method,
            learn_logvar_bounds,
            activation_fn_cfg,
        )

        self.in_size = in_size #Avoid dimension clashes in the assert n_dim = obs_space

    def forward(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x, y = torch.split(x, ceil(x.shape[-1] / 2), dim=-1)

        mean_x, log_var_x = super().forward(
            x, rng, propagation_indices, use_propagation
        )
        mean_y, log_var_y = super().forward(
            y, rng, propagation_indices, use_propagation
        )

        mean = torch.cat([mean_x, mean_y], dim=-1)
        log_var = torch.cat([log_var_x, log_var_y], dim=-1)

        return mean, log_var
