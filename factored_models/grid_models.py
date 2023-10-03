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

        self.factored_in_size = 2 #1 input for the current state, 1 input for the action
        self.factored_out_size = 1 #Might not work if we want to learn the reward function too !!!!

        super().__init__(
            self.factored_in_size,
            self.factored_out_size,
            device,
            num_layers,
            ensemble_size,
            hid_size,
            deterministic,
            propagation_method,
            learn_logvar_bounds,
            activation_fn_cfg,
        )

        # Avoid dimension clashes in the assert n_dim = obs_space
        self.in_size = in_size

    def create_factored_input(self, x):

        state, action = torch.split(x, self.in_size//2, dim=-1)

        states = torch.split(state, self.factored_in_size//2, dim=-1)
        actions = torch.split(action, self.factored_in_size//2, dim=-1)

        xs = []
        for state, action in zip(states, actions):
            x = torch.cat([state, action], dim=-1)
            xs.append(x)

        return xs


    def forward(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        xs = self.create_factored_input(x)

        for i, x in enumerate(xs):

            if i == 0:
                mean, log_var = super().forward(
                    x, rng, propagation_indices, use_propagation
                )
            else:
                mean_x, log_var_x = super().forward(
                    x, rng, propagation_indices, use_propagation
                )
                mean = torch.cat([mean, mean_x], dim=-1)
                log_var = torch.cat([log_var, log_var_x], dim=-1)

        return mean, log_var
