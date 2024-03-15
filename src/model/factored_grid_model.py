from typing import Any, Dict, Optional, Tuple, Union, List
from mbrl.types import ModelInput
import omegaconf
import torch

from mbrl.models.gaussian_mlp import GaussianMLP


# TODO: In a very general set-up, the "factored" sub-models should not be the same
# -> see factored_ffnn


class FactoredGaussianMLP(GaussianMLP):
    """
    FFNN factored GaussianMLP model
    Should be generalized as we move to more complex environments

    Instead of processing the whole state-action input tensor of dimension [state_space+state_action],
    It processes each factored state-action input of dimension [factored_in_size] such that the output
    is simply the concatenation of all the outputs of the factored state-action inputs.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        factored_in_size: int = 2,  # state, action
        factored_out_size: int = 1,  # next_state
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        self.factored_in_size = factored_in_size
        self.factored_out_size = factored_out_size
        self.factor = in_size // self.factored_in_size

        # "Factored submodels"
        super().__init__(
            self.factored_in_size,
            self.factored_out_size,
            device,
            num_layers,
            ensemble_size,
            hid_size // self.factor,
            deterministic,
            propagation_method,
            learn_logvar_bounds,
            activation_fn_cfg,
        )

        # Avoid dimension clashes in the assert n_dim = obs_space
        self.in_size = in_size

    def create_factored_input(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: whole state-action tensor
        :return: a list of factored state-action input tensors
        """

        state, action = torch.split(x, self.in_size // 2, dim=-1)

        states = torch.split(state, self.factored_in_size // 2, dim=-1)
        actions = torch.split(action, self.factored_in_size // 2, dim=-1)

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
