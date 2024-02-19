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
from src.model.simple import Simple, FactoredSimple




class MixtureModel(Model):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        factors: Optional[List] = None,
        reward_factors: Optional[List] = None,
        freq_train_reward: Optional[int] = 5,
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

        #TODO: Think about n_epochs, use diff learning rate etccc
        #pas grave si pour l'instant tout est hard codé pas propre

        Model.__init__(self, device)

        self.in_size = in_size
        self.out_size = out_size

        self.factors = factors
        self.reward_factors = reward_factors

        # In case we are learning the reward too
        self.learn_reward = True if self.out_size == len(self.factors) + 1 else False
        if self.learn_reward:
            if self.reward_factors is None:
                self.factors.append([i for i in range(self.in_size)])
            else:  # TODO: DOES NOT WORK FOR NOW
                raise ValueError("Factored reward not supported yet")
                self.reward_model_factors = self.get_model_factors(reward_factors)
                self.reward_models = self.get_factored_models(
                    hid_size=hid_size,
                    num_layers=num_layers,
                    activation_fn_cfg=activation_fn_cfg,
                    model_factors=self.reward_model_factors,
                )

        self.reward_model = FactoredMultiOutputGP(
            in_size,
            1,
            device,
            reward_factors,
            mean,
            kernel,
            scale_kernel,
        ) if reward_factors else MultiOutputGP(
            in_size,
            1,
            device,
            mean,
            kernel,
            scale_kernel,
        )
        self.dyn_model = FactoredSimple(
            in_size,
            out_size-1,
            device,
            factors,
            num_layers=num_layers,
            hid_size=hid_size,
            activation_fn_cfg=activation_fn_cfg,
        ) if factors else Simple(
            in_size,
            out_size-1,
            device,
            num_layers,
            hid_size,
            activation_fn_cfg,
        )
        
        self.apply(truncated_normal_init)
        self.to(self.device)
        self.freq_train_reward = freq_train_reward

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(x.shape) == 2

        batch_size = x.shape[0]
        pred = torch.empty(batch_size, self.out_size)
        pred[..., :-1] = self.dyn_model.forward(x)
        pred[..., -1] = self.reward_model.forward(x)
        return pred

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2

        batch_size = model_in.shape[0]
        dyn_eval_score, meta = self.dyn_model.eval_score(model_in, target)
        rew_eval_score, rew_meta = self.reward_model.eval_score(model_in, target)

        eval_score = torch.cat([dyn_eval_score, rew_eval_score], dim=-1)
        for key, value in meta:
            if key in rew_meta:
                value = torch.tensor([value, rew_meta[key]], dim=-1)

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
        dyn_loss, meta = self.dyn_model.update(model_in, optimizers[0], target, mode=mode)
        rew_loss, rew_meta = self.reward_model.update(model_in, optimizers[1], target)

        all_loss = np.concatenate([dyn_loss, rew_loss], axis=-1)
        for key, value in meta:
            if key in rew_meta:
                value = torch.tensor([value, rew_meta[key]], dim=-1)

        if mode == "mean":
            return np.mean(all_loss), meta
        elif mode == "separate":
            return all_loss, meta
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
            )
        
    def update_only_dynamics(
        self,
        model_in: ModelInput,
        optimizer: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        dyn_loss, meta = self.dyn_model.update(model_in, optimizer, target, mode='separate')
        rew_loss, rew_meta = self.reward_model.loss(model_in, target)

        all_loss = np.concatenate([dyn_loss, rew_loss], axis=-1)
        for key, value in meta:
            if key in rew_meta:
                value = torch.tensor([value, rew_meta[key]], dim=-1)

        if mode == "mean":
            return np.mean(all_loss), meta
        elif mode == "separate":
            return all_loss, meta
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
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
