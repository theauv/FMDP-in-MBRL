import pathlib
from typing import Any, Dict, Optional, Tuple, Union, List
from copy import deepcopy

import gpytorch
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
import torch
from torch.functional import F
from torch.optim.optimizer import Optimizer as Optimizer

from mbrl.models.model import Model


class ExactGPModel(gpytorch.models.ExactGP):
    # TODO: Reecrire cette classe avec plus de choix de kernel etc
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        mean: Optional[Mean] = None,
        kernel: Optional[Kernel] = None,
        scale_kernel: bool = True,
        in_size: Optional[int] = None,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean if mean is not None else gpytorch.means.ConstantMean()
        self.covar_module = (
            kernel if kernel is not None else gpytorch.kernels.RBFKernel()
        )
        if isinstance(self.mean_module, str):
            if self.mean_module == "Constant":
                self.mean_module = gpytorch.means.ConstantMean()
            elif self.mean_module == "Linear":
                if in_size is None:
                    raise ValueError("You chose a linear mean but the in_size is None")
                self.mean_module = gpytorch.means.LinearMean(in_size)
            else:
                raise ValueError(
                    f"No mean module named {self.mean_module}. You can added here if needed"
                )
        if isinstance(self.covar_module, str):
            if self.covar_module == "RBF":
                self.covar_module = gpytorch.kernels.RBFKernel()
            elif self.covar_module == "Matern":
                self.covar_module = gpytorch.kernels.MaternKernel()
            elif self.covar_module == "Linear":
                self.covar_module = gpytorch.kernels.LinearKernel()
            else:
                ValueError(
                    f"No kernel named {self.covar_module}. You can added here if needed"
                )
        if scale_kernel:
            self.covar_module = gpytorch.kernels.ScaleKernel(self.covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiOutputGP(Model):
    # TODO: make sure that the reset_target and input are fine (like that it is actually DONE and see
    # how to do it when in the mbrl pipelines)
    def __init__(
        self,
        in_size,
        out_size,
        device,
        mean: Optional[Mean] = None,
        kernel: Optional[Kernel] = None,
        scale_kernel: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(device, *args, **kwargs)

        self.in_size = in_size
        self.out_size = out_size
        models = []
        for i in range(out_size):
            models.append(
                ExactGPModel(
                    None,
                    None,
                    gpytorch.likelihoods.GaussianLikelihood(),
                    mean,
                    kernel,
                    scale_kernel,
                    in_size,
                )
            )

        self.gp = gpytorch.models.IndependentModelList(*models)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(
            *[submodel.likelihood for submodel in models]
        )
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.gp)

    def set_train_data(self, train_x=None, train_y=None, strict=False):
        for i, model in enumerate(self.gp.models):
            train_y_i = train_y[:, i]
            if model.train_inputs is not None:
                # TODO: Those assertions might not even be useful if we have a
                # model that updates its factors over time... (So not useful in general)
                assert train_x.shape[-1] == model.train_inputs[0].shape[-1]
                assert train_y_i.ndim == model.train_targets.ndim == 1
            model.set_train_data(train_x, train_y_i, strict=strict)

    def forward(
        self, x: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x is None:
            return self.gp(*self.gp.train_inputs)
        else:
            from time import time
            start = time()
            x = x.unsqueeze(0).repeat(self.out_size, 1, 1)
            loss = self.gp(*x)
            return loss

    def loss(self, model_in: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        # TODO: might be avoidably very computationally expensive !!
        self.set_train_data(model_in, target)
        pred_out = self.forward()
        # TODO Debug: Be sure target is well passed before in the GP as it is NOT USED
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_mean = self.likelihood(*pred_out)
            pred_mean = torch.cat(
                [pred.mean.unsqueeze(-1) for pred in pred_mean], axis=-1
            )
            meta = {
                "outputs": pred_mean,
                "targets": target,
            }
        return -self.mll(pred_out, self.gp.train_targets), meta

    def pred_distribution(
        self, model_in: torch.Tensor
    ) -> torch.distributions.Distribution:
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self.likelihood(*self.forward(model_in))

    def eval_score(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # if self.models.train_inputs ar all None -> None
        assert model_in.ndim == 2 and target.ndim == 2
        pred_output = self.pred_distribution(model_in)
        pred_mean = torch.cat(
            [pred.mean.unsqueeze(-1) for pred in pred_output], axis=-1
        )
        meta = {
            "outputs": pred_mean,
            "targets": target,
        }
        return F.mse_loss(pred_mean, target, reduction="none").unsqueeze(0), meta

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {"state_dict": [model.state_dict() for model in self.gp.models]}
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        for i, model in enumerate(self.gp.models):
            model.load_state_dict(model_dict["state_dict"][i])
        self.gp = gpytorch.models.IndependentModelList(*self.gp.models)

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
        # Need to think about the advantage of having a distribution instead of a prediction
        pred_output = self.pred_distribution(model_input)
        pred_mean = torch.cat(
            [pred.mean.unsqueeze(-1) for pred in pred_output], axis=-1
        )
        return (pred_mean, model_state)


class FactoredMultiOutputGP(MultiOutputGP):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        factors: List,
        mean: Optional[Mean] = None,
        kernel: Optional[Kernel] = None,
        scale_kernel: bool = True,
        reward_factors: Optional[List] = None,
    ):
        """
        :param raw_factors: Adjacency list for which each entry i is a tuple or a List of the inputs the output i
        depends on
        """

        Model.__init__(self, device)

        self.mean = mean
        self.kernel = kernel
        self.scale_kernel = scale_kernel

        self.in_size = in_size
        self.out_size = out_size

        self.factors = factors
        self.reward_factors = reward_factors

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
        self.models = self.get_factored_models()

        self.gp = gpytorch.models.IndependentModelList(*self.models)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(
            *[submodel.likelihood for submodel in self.models]
        )
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.gp)

    @staticmethod
    def get_model_factors(raw_factors_):
        """Compute the scope (input) for each single factor (output)"""
        raw_factors = deepcopy(raw_factors_)
        factors = []
        for output, inputs in enumerate(raw_factors):
            inputs = list(inputs)
            out = [output]
            factors.append((inputs, out))
        return factors

    def get_factored_models(
        self,
    ):
        models = []
        total_out_size = 0
        for model_factor in self.model_factors:
            assert len(model_factor) == 2
            in_size = len(model_factor[0])
            total_out_size += 1

            model = ExactGPModel(
                None,
                None,
                gpytorch.likelihoods.GaussianLikelihood(),
                self.mean,
                self.kernel,
                self.scale_kernel,
                in_size,
            )
            models.append(model)

        print(total_out_size, self.out_size)
        assert total_out_size == self.out_size
        return models

    def set_train_data(self, train_x=None, train_y=None, strict=False):
        for i, model in enumerate(self.gp.models):
            input_idx, output_idx = self.model_factors[i]
            train_y_i = train_y[:, i]
            train_x_i = train_x.index_select(-1, torch.tensor(self.model_factors[i][0]))
            if model.train_inputs is not None:
                # TODO: Those assertions might not even be useful if we have a
                # model that updates its factors over time... (So not useful in general)
                assert train_x_i.shape[-1] == model.train_inputs[0].shape[-1]
                assert train_y_i.ndim == model.train_targets.ndim == 1
            model.set_train_data(train_x_i, train_y_i, strict=strict)

    def forward(
        self, x: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x is None:
            return self.gp(*self.gp.train_inputs)
        else:
            assert len(self.gp.models) == len(self.model_factors)
            out = []
            for i, model in enumerate(self.gp.models):
                sub_x = x.index_select(-1, torch.tensor(self.model_factors[i][0]))
                out.append(model(sub_x))
            return out
