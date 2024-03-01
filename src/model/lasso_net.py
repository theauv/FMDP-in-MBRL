from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import warnings

import hydra
import numpy as np
import omegaconf
import torch
from torch import nn
from torch.functional import F

from mbrl.models.model import Model
from mbrl.models.util import truncated_normal_init
from mbrl.types import ModelInput

from lassonet import LassoNet

from src.model.simple import Simple


class LassoNetAdapted(LassoNet):
    """
    LassoNet easily usable with the mbrl library set-up of the project
    TODO: MAKE USE OF GROUPS ???? IS IT USEFUL ??
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]],
        num_layers: int,
        hid_size: int,
        groups: Optional[int] = None,
        dropout: Optional[bool] = None,
    ):
        self.in_size = in_size
        self.out_size = out_size

        if groups is not None:
            n_inputs = in_size
            all_indices = []
            for g in groups:
                for i in g:
                    all_indices.append(i)
            assert len(all_indices) == n_inputs and set(all_indices) == set(
                range(n_inputs)
            ), f"Groups must be a partition of range(n_inputs={n_inputs})"

        self.groups = groups

        nn.Module.__init__(self)

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        layers = [nn.Linear(in_size, hid_size)]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hid_size, hid_size))
        layers.append(nn.Linear(hid_size, out_size))

        self.action_layer = create_activation()
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.skip = nn.Linear(in_size, out_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_no_grad = hasattr(self, "mask")
        current_layer = x
        if skip_no_grad:
            with torch.no_grad():
                result = self.skip(x)
        else:
            result = self.skip(x)
        for i, theta in enumerate(self.layers):
            if i == 0 and skip_no_grad:
                theta.weight = nn.Parameter(
                    torch.where(self.mask, theta.weight, torch.tensor(0.0))
                )
                current_layer = theta(current_layer)
            else:
                current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = self.action_layer(current_layer)
        return current_layer + result

    def create_mask(self, mask):
        self.register_buffer("mask", mask)


class LassoSimple(Simple):
    """
    This Model takes all its sense to be use with the associated LassoModelTrainer (and Wrapper).
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        groups: Optional[int] = None,
        dropout: bool = False,
        gamma: float = 0.0,
        gamma_skip: float = 0.0,
        M: float = 1.0,
    ):
        Model.__init__(self, device)

        self.in_size = in_size
        self.out_size = out_size
        self.gamma = gamma
        self.gamma_skip = gamma_skip
        self.M = M
        self.num_layers = num_layers
        self.hid_size = hid_size
        self.activation_fn_cfg = activation_fn_cfg
        self.factors = None

        self.lassonets = []
        for i in range(self.out_size):
            lassonet = LassoNetAdapted(
                in_size, 1, activation_fn_cfg, num_layers, hid_size, groups, dropout
            )
            self.lassonets.append(lassonet)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def set_factors(self, factors):
        self.factors = factors

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for i, lassonet in enumerate(self.lassonets):
            if i == 0:
                pred = lassonet.forward(x)
            else:
                next_pred = lassonet.forward(x)
                pred = torch.cat([pred, next_pred], dim=-1)
        return pred

    def lassonet_loss(
        self,
        lassonet: LassoNetAdapted,
        model_in: torch.Tensor,
        target: torch.Tensor = None,
    ) -> torch.Tensor:
        assert model_in.ndim == 2 and target.ndim == 2  # Not sure
        assert target.shape[-1] == 1

        pred_out = lassonet.forward(model_in)
        meta = {"outputs": pred_out, "targets": target}
        return F.mse_loss(pred_out, target, reduction="none").mean(-1).mean(), meta

    def lassonet_eval_score(
        self,
        lassonet: LassoNetAdapted,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        assert target.shape[-1] == 1

        if lambda_ is None:
            warnings.warn("You did not give any lambda, default lambda = 0.")
            lambda_ = 0.0

        with torch.no_grad():
            pred_out = lassonet.forward(model_in)
            loss = F.mse_loss(pred_out, target, reduction="none")
            loss = (
                loss  # .item()
                + lambda_ * lassonet.l1_regularization_skip().item()
                + self.gamma * lassonet.l2_regularization().item()
                + self.gamma_skip * lassonet.l2_regularization_skip().item()
            ).unsqueeze(0)
            meta = {"outputs": pred_out, "targets": target}
        return loss, meta

    def eval_score(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if lambda_ is None:
            warnings.warn("You did not give any lambda, default lambda = 0.")
            lambda_ = 0.0

        assert model_in.ndim == 2 and target.ndim == 2
        assert target.shape[-1] == len(self.lassonets)

        eval_scores = []
        metas = {}
        # Compute the loss for each single output and its associated lassonet model
        for i, lassonet in enumerate(self.lassonets):
            sub_target = target[:, i].unsqueeze(-1)
            eval_score, meta = self.lassonet_eval_score(
                lassonet, model_in, sub_target, lambda_
            )
            eval_scores.append(eval_score)
            if i == 0:
                metas = meta
            else:
                metas = {
                    key: torch.cat([value, meta[key]], dim=-1)
                    for key, value in metas.items()
                }

        n_outputs = len(eval_scores)
        assert n_outputs == self.out_size

        return torch.cat(eval_scores, dim=-1), metas

    def lassonet_update(
        self,
        lassonet: LassoNetAdapted,
        model_in: ModelInput,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[float, Dict[str, Any]]:
        if lambda_ is None:
            self.train()
            optimizer.zero_grad()
            loss, meta = self.lassonet_loss(lassonet, model_in, target)
            loss.backward()
            if meta is not None:
                with torch.no_grad():
                    grad_norm = 0.0
                    for p in list(
                        filter(lambda p: p.grad is not None, self.parameters())
                    ):
                        grad_norm += p.grad.data.norm(2).item() ** 2
                    meta["grad_norm"] = grad_norm
            optimizer.step()
            return loss.item(), meta

        def closure():
            optimizer.zero_grad()
            loss, meta = self.lassonet_loss(lassonet, model_in=model_in, target=target)
            ans = (
                # TODO: Might not be a good idea full_loss + hiddenlayers reg
                loss
                + self.gamma * lassonet.l2_regularization()
                + self.gamma_skip * lassonet.l2_regularization_skip()
            )

            # TODO Keep these prints ??
            if ans + 1 == ans:
                print(f"Loss is {ans}", file=sys.stderr)
                print(f"Did you normalize input?", file=sys.stderr)
                print(f"Loss: {loss}")
                print(f"l2_regularization: {lassonet.l2_regularization()}")
                print(f"l2_regularization_skip: {lassonet.l2_regularization_skip()}")
                assert False
            ans.backward()
            if meta is not None:
                with torch.no_grad():
                    grad_norm = 0.0
                    for p in list(
                        filter(lambda p: p.grad is not None, self.parameters())
                    ):
                        grad_norm += p.grad.data.norm(2).item() ** 2
                    meta["grad_norm"] = grad_norm
            return ans, meta

        ans, meta = optimizer.step(closure)
        lassonet.prox(lambda_=lambda_ * optimizer.param_groups[0]["lr"], M=self.M)

        return ans.item(), meta

    def update(
        self,
        model_in: ModelInput,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2

        self.train()

        all_ans = []
        metas = {}
        for i, lassonet in enumerate(self.lassonets):
            sub_target = target[:, i].unsqueeze(-1)
            ans, meta = self.lassonet_update(
                lassonet,
                model_in,
                optimizer=optimizers[i],
                target=sub_target,
                lambda_=lambda_,
            )
            all_ans.append(ans)
            if i == 0:
                metas = meta
            else:
                metas = {
                    key: torch.cat([value, meta[key]], dim=-1)
                    if hasattr(value, "__len__")
                    else value + meta[key]
                    for key, value in metas.items()
                }


        n_outputs = len(all_ans)
        assert n_outputs == self.out_size
        if mode == "mean":
            return np.mean(all_ans), metas
        elif mode == "separate":
            return all_ans, metas
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
            )
