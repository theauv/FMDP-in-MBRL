from copy import deepcopy
from math import ceil
from typing import Union, Optional, Dict, Tuple, Any, List

import hydra
from mbrl.types import ModelInput
import numpy as np
import omegaconf
import pathlib
import torch
from torch import nn
from torch.functional import F

from mbrl.models.model import Model
from mbrl.models.util import truncated_normal_init


class Simple(Model):
    """
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

        self.factors = [np.arange(self.in_size) for output in range(self.out_size)]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.hidden_layers(x)

    def loss(self, model_in: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        assert model_in.ndim == 2 and target.ndim == 2
        pred_out = self.forward(model_in)
        return F.mse_loss(pred_out, target, reduction="none").mean(-1).mean(), {}

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_output = self.forward(model_in)
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

        self.factors = [
            [i, i + self.out_size] for i, output in enumerate(range(self.out_size))
        ]

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


class FactoredSimple(Simple):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        factors: List,
        reward_factors: Optional[List] = None,
        num_layers: int = 4,
        hid_size: int = 200,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        """
        :param raw_factors: Adjacency list for which each entry i is a tuple or a List of the inputs the output i
        depends on
        """

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

        self.model_factors = self.get_model_factors(self.factors)
        self.models = self.get_factored_models(
            hid_size=hid_size,
            num_layers=num_layers,
            activation_fn_cfg=activation_fn_cfg,
            model_factors=self.model_factors,
        )

        self.apply(truncated_normal_init)
        self.to(self.device)

    @staticmethod
    def get_model_factors(raw_factors_):
        """_summary_

        :param raw_factors: List for which each entry i is a tuple or a List of the inputs the output i
        depends on
        :return: The Linked list of the factored models: [(in_1, out_1), ..., (in_n, out_n)]. in_i is the inputs
        playing in a role in the model i, out_i the outputs of the model i.
        """
        raw_factors = deepcopy(raw_factors_)
        factors = []
        for output, inputs in enumerate(raw_factors):
            inputs = list(inputs)
            out = [output]
            j = output + 1
            while j < len(raw_factors):
                other_inputs = list(raw_factors[j])
                if inputs == other_inputs:
                    out.append(j)
                    raw_factors.pop(j)
                j += 1
            factors.append((inputs, out))
        return factors

    def get_factored_models(
        self, hid_size, num_layers, activation_fn_cfg, model_factors
    ):
        models = []
        total_out_size = 0
        for model_factor in model_factors:
            assert len(model_factor) == 2
            in_size = len(model_factor[0])
            out_size = len(model_factor[1])
            total_out_size += out_size

            # TODO: Check this operation makes sense
            if hid_size > in_size + out_size:
                reduction = max(in_size / self.in_size, out_size / self.out_size)
                new_hid_size = ceil(hid_size * reduction)

            model = Simple(
                in_size=in_size,
                out_size=out_size,
                num_layers=num_layers,
                hid_size=new_hid_size,
                activation_fn_cfg=activation_fn_cfg,
                device=self.device,
            )
            models.append(model)

        assert total_out_size == self.out_size
        return models

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert len(x.shape) == 2

        batch_size = x.shape[0]
        preds = torch.empty(batch_size, self.out_size)
        for i, model_i in enumerate(self.models):
            input_idx, output_idx = self.model_factors[i]
            sub_x = x.index_select(-1, index=torch.tensor(input_idx))
            pred = model_i.forward(sub_x)
            preds[:, output_idx] = pred

        # In case we are learning the factors too
        if self.learn_reward and self.reward_factors is not None:
            for i, model_i in enumerate(self.reward_models):
                input_idx, output_idx = self.reward_model_factors[i]
                sub_x = x.index_select(-1, index=torch.tensor(input_idx))
                pred = model_i.forward(sub_x)
                preds[:, -1] += pred

        return preds

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2

        batch_size = model_in.shape[0]
        eval_scores = torch.zeros((1, batch_size, self.out_size))
        metas = []
        # Compute the loss for each single output and its associated lassonet model
        for i, model in enumerate(self.models):
            sub_model_in = model_in.index_select(
                -1, torch.tensor(self.model_factors[i][0])
            )
            sub_target = target.index_select(-1, torch.tensor(self.model_factors[i][1]))
            eval_score, meta = model.eval_score(sub_model_in, sub_target)
            eval_scores[0, :, self.model_factors[i][1]] = eval_score
            metas.append(meta)

        if self.learn_reward and self.reward_factors is not None:
            pred_reward = 0
            reward_target = target[:, -1]
            for i, model in enumerate(self.reward_models):
                sub_model_in = model_in.index_select(
                    -1, torch.tensor(self.reward_model_factors[i][0])
                )
                assert sub_model_in.ndim == 2 and sub_target.ndim == 2
                with torch.no_grad():
                    pred_reward += model.forward(sub_model_in)

            eval_score, meta = (
                F.mse_loss(pred_reward, reward_target, reduction="none"),
                {},
            )
            eval_scores[0, :, -1] = eval_score
            metas.append(meta)

        return eval_scores, {}

    def update(
        self,
        model_in: ModelInput,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        mode: str = "mean",
    ) -> Tuple[float, Dict[str, Any]]:
        assert model_in.ndim == 2 and target.ndim == 2

        self.train()

        all_loss = []
        all_meta = []
        for i, model in enumerate(self.models):
            sub_model_in = model_in.index_select(
                -1, torch.tensor(self.model_factors[i][0])
            )
            sub_target = target.index_select(-1, torch.tensor(self.model_factors[i][1]))

            loss, meta = model.update(
                sub_model_in, optimizer=optimizers[i], target=sub_target
            )
            all_loss.append(loss)
            all_meta.append(meta)

        if self.learn_reward and self.reward_factors is not None:
            # TODO: Does not work either here
            raise ValueError("Factored reward not supported yet")
            for i, model in enumerate(self.reward_models):
                pred_reward = 0
                reward_target = target[:, -1]
                sub_model_in = model_in.index_select(
                    -1, torch.tensor(self.reward_model_factors[i][0])
                )
                loss, meta = model.update(
                    sub_model_in, optimizer=optimizers[i], target=reward_target
                )
                all_loss.append(loss)
                all_meta.append(meta)

        if mode == "mean":
            return np.mean(all_loss), {}
        elif mode == "separate":
            return all_loss, all_meta
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
            )
