import pathlib
from typing import Union, Optional, Dict, Any, Tuple

import torch
from torch.functional import F

from mbrl.models import Model


class LinearRegression(Model):
    def __init__(self, in_size: int, out_size: int, device: Union[str, torch.device], logistic: bool = False):
        super().__init__(device)
        self.linear = torch.nn.Linear(in_size, out_size)
        self.criterion = torch.nn.MSELoss()
        self.in_size = in_size
        self.out_size = out_size
        self.logistic = logistic

    def forward(self, x):
        x = self.linear(x)
        if self.logistic:
            return torch.sigmoid(x)
        else:
            return x

    def loss(
        self, model_in: torch.Tensor, target: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        pred_out = self.forward(model_in)
        meta = {"outputs": pred_out, "targets": target}
        return self.criterion(pred_out, target), meta

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_output = self.forward(model_in)
            meta = {"outputs": pred_output, "targets": target}
            return F.mse_loss(pred_output, target, reduction="none").unsqueeze(0), meta

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {"state_dict": self.state_dict()}
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])

    def reset_1d(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        assert rng is not None
        propagation_indices = None
        return {"obs": obs, "propagation_indices": propagation_indices}

    def sample_1d(
        self,
        model_input: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        return (self.forward(model_input), model_state)
