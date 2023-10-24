# TODO: Deprecated, has to be deleted

# import sys
# from typing import Any, Dict, Optional, Tuple, Union
# import warnings

# import omegaconf
# import torch
# from torch.nn import functional as F
# from torch import optim as optim

# from mbrl.types import ModelInput
# from mbrl.models.gaussian_mlp import GaussianMLP
# from lassonet.model import LassoNet


# class LassoNetGaussianMLP(GaussianMLP):
#     """
#     Simple factored GaussianMLP model
#     Should be generalized as we move to more complex environments

#     Instead of processing the whole state-action input tensor of dimension [state_space+state_action],
#     It processes each factored state-action input of dimension [factored_in_size] such that the output
#     is simply the concatenation of all the outputs of the factored state-action inputs.
#     """

#     def __init__(
#         self,
#         in_size: int,
#         out_size: int,
#         device: Union[str, torch.device],
#         num_layers: int = 4,
#         ensemble_size: int = 1,
#         hid_size: int = 200,
#         deterministic: bool = False,
#         propagation_method: Optional[str] = None,
#         learn_logvar_bounds: bool = False,
#         activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
#         gamma: float = 0.0,
#         gamma_skip: float = 0.0,
#         M: float = 10.0,
#     ):

#         super().__init__(
#             in_size,
#             out_size,
#             device,
#             num_layers,
#             ensemble_size,
#             hid_size,
#             deterministic,
#             propagation_method,
#             learn_logvar_bounds,
#             activation_fn_cfg,
#         )

#         self.hidden_dims = tuple([hid_size for layer in range(num_layers)])

#         self.hidden_layers = LassoNet(in_size, *self.hidden_dims)
#         self.gamma = gamma
#         self.gamma_skip = gamma_skip
#         self.M = M

#     def eval_score(  # type: ignore
#         self,
#         model_in: torch.Tensor,
#         target: Optional[torch.Tensor] = None,
#         lambda_: float = None,
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:

#         if lambda_ is None:
#             warnings.warn("You are not evaluating using the lasso solution")
#             return super().eval_score(model_in, target)

#         assert model_in.ndim == 2 and target.ndim == 2
#         with torch.no_grad():
#             pred_mean, _ = self.forward(model_in, use_propagation=False)
#             target = target.repeat((self.num_members, 1, 1))
#             loss = F.mse_loss(pred_mean, target, reduction="none")
#             return (
#                 (
#                     loss  # .item()
#                     + lambda_ * self.hidden_layers.l1_regularization_skip().item()
#                     + self.gamma * self.hidden_layers.l2_regularization().item()
#                     + self.gamma_skip
#                     * self.hidden_layers.l2_regularization_skip().item()
#                 ),
#                 {},
#             )

#     def update(
#         self,
#         model_in: ModelInput,
#         optimizer: torch.optim.Optimizer,
#         target: Optional[torch.Tensor] = None,
#         lambda_: float = None,
#     ) -> Tuple[float, Dict[str, Any]]:

#         # TODO: Necessary ??
#         if lambda_ is None:
#             warnings.warn("You are not updating the model using Lasso")
#             return super().update(model_in, optimizer, target)

#         self.train()

#         def closure():
#             optimizer.zero_grad()
#             loss, meta = self.loss(model_in, target)
#             ans = (
#                 # TODO: Might not be a good idea full_loss + hiddenlayers reg
#                 loss
#                 + self.gamma * self.hidden_layers.l2_regularization()
#                 + self.gamma_skip * self.hidden_layers.l2_regularization_skip()
#             )

#             # TODO Keep these prints ??
#             if ans + 1 == ans:
#                 print(f"Loss is {ans}", file=sys.stderr)
#                 print(f"Did you normalize input?", file=sys.stderr)
#                 print(f"Loss: {loss}")
#                 print(f"l2_regularization: {self.hidden_layers.l2_regularization()}")
#                 print(
#                     f"l2_regularization_skip: {self.hidden_layers.l2_regularization_skip()}"
#                 )
#                 assert False
#             ans.backward()
#             if meta is not None:
#                 with torch.no_grad():
#                     grad_norm = 0.0
#                     for p in list(
#                         filter(lambda p: p.grad is not None, self.parameters())
#                     ):
#                         grad_norm += p.grad.data.norm(2).item() ** 2
#                     meta["grad_norm"] = grad_norm
#             return ans, meta

#         ans, meta = optimizer.step(closure)

#         # TODO: Useful ??
#         self.hidden_layers.prox(
#             lambda_=lambda_ * optimizer.param_groups[0]["lr"], M=self.M
#         )

#         return ans.item(), meta
