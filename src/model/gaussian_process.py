import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import gpytorch
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from mbrl.types import ModelInput
import torch
from torch._tensor import Tensor
from torch.functional import F
from torch.optim.optimizer import Optimizer as Optimizer
from torcheval.metrics.functional import r2_score

from mbrl.models.model import Model

from typing import Any, Dict, Optional, Tuple, Union
from torch.functional import F
import pathlib

from mbrl.models.model import Model

from time import time


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
            if self.mean_module=="Constant":
                self.mean_module=gpytorch.means.ConstantMean()
            elif self.mean_module=="Linear":
                if in_size is None:
                    raise ValueError("You chose a linear mean but the in_size is None")
                self.mean_module=gpytorch.means.LinearMean(in_size)
            else:
                raise ValueError(f"No mean module named {self.mean_module}. You can added here if needed")
        if isinstance(self.covar_module, str):
            if self.covar_module=="RBF":
                self.covar_module=gpytorch.kernels.RBFKernel()
            elif self.covar_module=="Matern":
                self.covar_module=gpytorch.kernels.MaternKernel()
            elif self.covar_module=="Linear":
                self.covar_module=gpytorch.kernels.LinearKernel()
            else:
                ValueError(f"No kernel named {self.covar_module}. You can added here if needed")
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
        eval_metric: Optional[str] = "MSE",
        *args,
        **kwargs
    ):
        super().__init__(device, *args, **kwargs)

        self.eval_metric = eval_metric
        self.in_size = in_size
        self.out_size = out_size
        models = []
        for i in range(out_size):
            models.append(
                ExactGPModel(
                    None, None, gpytorch.likelihoods.GaussianLikelihood(), mean, kernel, scale_kernel, in_size
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
            return [model(x) for model in self.gp.models]

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
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.eval()  # Here or outside ??
            return self.likelihood(*self.forward(model_in))

    def eval_score(
        self, model_in: torch.Tensor, target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        #if self.models.train_inputs ar all None -> None
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
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
