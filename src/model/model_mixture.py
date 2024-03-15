from typing import Union, Optional, Dict, Tuple, Any, List

from mbrl.types import ModelInput
import numpy as np
import omegaconf
import torch
from gpytorch.means import Mean
from gpytorch.kernels import Kernel

from mbrl.models.model import Model
from mbrl.models.util import truncated_normal_init

from src.model.gaussian_process import MultiOutputGP, FactoredMultiOutputGP
from src.model.neural_network import FFNN, FactoredFFNN


class MixtureModel(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        factors: Optional[List] = None,
        reward_factors: Optional[List] = None,
        freq_train_reward: Optional[int] = 10,
        num_layers: int = 4,
        hid_size: int = 200,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        mean: Optional[Mean] = None,
        kernel: Optional[Kernel] = None,
        scale_kernel: bool = True,
    ):
        """
        :param raw_factors: Adjacency list for which each entry i is a tuple or a List of the inputs the output i
        depends on
        """

        # TODO: Think about n_epochs, use diff learning rate etccc
        # pas grave si pour l'instant tout est hard codé pas propre

        Model.__init__(self, device)

        self.in_size = in_size
        self.out_size = out_size

        self.factors = factors
        self.reward_factors = reward_factors

        self.reward_model = (
            FactoredMultiOutputGP(
                in_size, 1, device, reward_factors, mean, kernel, scale_kernel
            )
            if reward_factors
            else MultiOutputGP(in_size, 1, device, mean, kernel, scale_kernel)
        )
        self.dyn_model = (
            FactoredFFNN(
                in_size,
                out_size - 1,
                device,
                factors,
                num_layers=num_layers,
                hid_size=hid_size,
                activation_fn_cfg=activation_fn_cfg,
            )
            if factors
            else FFNN(
                in_size, out_size - 1, device, num_layers, hid_size, activation_fn_cfg
            )
        )

        self.apply(truncated_normal_init)
        self.to(self.device)
        self.freq_train_reward = freq_train_reward

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(x.shape) == 2

        batch_size = x.shape[0]
        pred = torch.empty(batch_size, self.out_size)
        pred[..., :-1] = self.dyn_model.forward(x)
        pred_reward = self.reward_model.pred_distribution(x)
        pred[..., -1] = pred_reward[0].mean
        return pred

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2

        batch_size = model_in.shape[0]
        dyn_eval_score, meta = self.dyn_model.eval_score(model_in, target[..., :-1])
        rew_eval_score, rew_meta = self.reward_model.eval_score(
            model_in, target[..., -1][..., None]
        )

        eval_score = torch.cat([dyn_eval_score, rew_eval_score], dim=-1)
        for key, value in meta.items():
            if key in rew_meta:
                value = torch.cat([value, rew_meta[key]], dim=-1)

        return eval_score, meta

    def update_all(
        self,
        model_in: ModelInput,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        assert len(optimizers) == 2

        batch_size = model_in.shape[0]
        all_loss, meta = self.dyn_model.update(
            model_in, optimizers[0], target[..., :-1], mode="separate"
        )
        rew_loss, rew_meta = self.reward_model.update(
            model_in, optimizers[1], target[..., -1][..., None]
        )

        all_loss.append(rew_loss)
        for key, value in meta.items():
            if key in rew_meta:
                if torch.is_tensor(value):
                    value = torch.cat([value, rew_meta[key]], dim=-1)
                else:
                    value += rew_meta[key]

        if mode == "mean":
            return np.mean(all_loss), meta
        elif mode == "separate":
            return all_loss, meta
        else:
            raise ValueError(
                f"There is no {mode} mode for the FFNNLasso eval_score method"
            )

    def update_only_dynamics(
        self,
        model_in: ModelInput,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        all_loss, meta = self.dyn_model.update(
            model_in, optimizers[0], target[..., :-1], mode="separate"
        )
        rew_loss, rew_meta = self.reward_model.loss(
            model_in, target[..., -1][..., None]
        )

        all_loss.append(rew_loss.detach().item())
        for key, value in meta.items():
            if key in rew_meta:
                value = torch.cat([value, rew_meta[key]], dim=-1)

        if mode == "mean":
            return np.mean(all_loss), meta
        elif mode == "separate":
            return all_loss, meta
        else:
            raise ValueError(
                f"There is no {mode} mode for the FFNNLasso eval_score method"
            )

    def update(
        self,
        model_in: ModelInput,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:
        for i in range(self.freq_train_reward):
            loss, meta = self.update_only_dynamics(model_in, optimizers, target, mode)
        return self.update_all(model_in, optimizers, target, mode)

    def loss(self, model_in: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        assert model_in.ndim == 2 and target.ndim == 2
        dyn_loss, meta = self.dyn_model.loss(model_in, target)
        rew_loss, rew_meta = self.reward_model.loss(model_in, target)
        loss = np.mean([dyn_loss, rew_loss])
        for key, value in meta.items():
            if key in rew_meta:
                value = torch.cat([value, rew_meta[key]], dim=-1)
        return loss, meta

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
