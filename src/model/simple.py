from math import ceil
from typing import Union, Optional, Dict, Tuple, Any, List
import sys
import warnings

import hydra
from mbrl.types import ModelInput
import omegaconf
import pathlib
import torch
from torch import nn
from torch.functional import F

from mbrl.models.model import Model
from mbrl.models.util import truncated_normal_init
from src.model.lasso_net import LassoNetAdapted


class Simple(Model):
    """
    TODO: Propagation indices and deterministic ???
    Simplest model supported by mbrl-lib
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(device)

        self.in_size = in_size
        self.out_size = out_size

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        hidden_layers = [
            nn.Sequential(nn.Linear(in_size, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(nn.Linear(hid_size, hid_size), create_activation())
            )
        hidden_layers.append(nn.Linear(hid_size, out_size))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.hidden_layers(x)

    def loss(self, model_in: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_out = self.forward(model_in)
        return F.mse_loss(pred_out, target, reduction="none").sum((1, 2)).sum(), {}

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_output = self.forward(model_in)
            target = target.repeat((1, 1, 1))  # Why
            return F.mse_loss(pred_output, target, reduction="none"), {}

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


class GridFactoredSimple(Simple):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        hid_size: int = 200,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):

        self.n_dims = out_size
        self.factored_in_size = in_size // self.n_dims
        self.factored_out_size = 1
        if hid_size > self.n_dims:
            hid_size = ceil(hid_size / self.n_dims)

        super().__init__(
            self.factored_in_size,
            self.factored_out_size,
            device,
            num_layers,
            hid_size,
            activation_fn_cfg,
        )

        self.in_size = in_size
        self.out_size = out_size

    def create_factored_input(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: whole state-action tensor
        :return: a list of factored state-action input tensors
        """

        state, action = torch.split(x, self.in_size // 2, dim=-1)

        states = torch.split(state, 1, dim=-1)
        actions = torch.split(action, 1, dim=-1)

        xs = []
        for state, action in zip(states, actions):
            x = torch.cat([state, action], dim=-1)
            xs.append(x)

        return xs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        xs = self.create_factored_input(x)
        for i, factored_x in enumerate(xs):
            if i == 0:
                pred = super().forward(factored_x)
            else:
                next_pred = super().forward(factored_x)
                pred = torch.cat([pred, next_pred], dim=-1)
        return pred


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
        M: float = 10.0,
    ):

        Model.__init__(self, device)  # If does not work try super().super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.gamma = gamma
        self.gamma_skip = gamma_skip
        self.M = M

        self.lassonets = []
        for i in range(self.out_size):
            lassonet = LassoNetAdapted(
                in_size, 1, activation_fn_cfg, num_layers, hid_size, groups, dropout
            )
            self.lassonets.append(lassonet)

        self.apply(truncated_normal_init)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        for i, lassonet in enumerate(self.lassonets):
            if i == 0:
                pred = lassonet.forward(x)
            else:
                next_pred = lassonet.forward(x)
                pred = torch.cat([pred, next_pred], dim=-1)
        return pred

    def lassonet_eval_score(
        self,
        lassonet: LassoNetAdapted,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

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
            )
            meta = {}

        return loss, meta

    def eval_score(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
        mode: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if lambda_ is None:
            warnings.warn("You did not give any lambda, default lambda = 0.")
            lambda_ = 0.0

        assert model_in.ndim == 2 and target.ndim == 2

        losses = []
        metas = []
        # Compute the loss for each single output and its associated lassonet model
        for i, lassonet in enumerate(self.lassonets):
            target = target.repeat((1, 1, 1))[:, :, i]
            loss, meta = self.lassonet_eval_score(lassonet, model_in, target, lambda_)
            losses.append(loss)
            metas.append(meta)

        if mode == "mean":
            n_outputs = len(losses)
            assert n_outputs == self.out_size
            return sum(losses) / n_outputs, {}
        elif mode == "separate":
            return losses, metas
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
            )

    def lassonet_update(
        self,
        lassonet: LassoNetAdapted,
        model_in: ModelInput,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[float, Dict[str, Any]]:

        if lambda_ is None:
            warnings.warn("You did not give any lambda, default lambda = 0.")
            lambda_ = 0.0

        def closure():
            optimizer.zero_grad()
            loss, meta = self.loss(model_in, target)
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
                print(f"l2_regularization: {self.hidden_layers.l2_regularization()}")
                print(
                    f"l2_regularization_skip: {self.hidden_layers.l2_regularization_skip()}"
                )
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

        return ans, meta

    def update(
        self,
        model_in: ModelInput,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:

        if lambda_ is None:
            warnings.warn("You did not give any lambda, default lambda = 0.")
            lambda_ = 0.0

        assert model_in.ndim == 2 and target.ndim == 2

        self.train()

        all_ans = []
        all_meta = []
        for i, lassonet in enumerate(self.lassonets):

            sub_target = target[:, i].unsqueeze(-1)

            ans, meta = self.lassonet_update(
                lassonet,
                model_in,
                optimizer=optimizers[i],
                target=sub_target,
                lambda_=lambda_,
            )
            all_ans.append(ans.item())
            all_meta.append(meta)

        if mode == "mean":
            n_outputs = len(all_ans)
            assert n_outputs == self.out_size
            return sum(all_ans) / n_outputs, {}
        elif mode == "separate":
            return all_ans, all_meta
