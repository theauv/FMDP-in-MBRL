from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from math import ceil
import sys
import tqdm
import warnings

import hydra
from matplotlib import pyplot as plt
import numpy as np
import omegaconf
import torch
from torch import optim
from torch import nn
from torch.functional import F

import mbrl
from mbrl.models.model import Model
from mbrl.models.model_trainer import ModelTrainer, MODEL_LOG_FORMAT
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.models.util import truncated_normal_init
from mbrl.types import ModelInput
from mbrl.util.replay_buffer import TransitionIterator, BootstrapIterator
from mbrl.util.logger import Logger

from lassonet import LassoNet

from src.model.simple import Simple, FactoredSimple


SPARSITY_LOG_FORMAT = [
    ("lambda", "L", "float"),
    ("lambda_iteration", "LI", "int"),
    ("epoch", "E", "int"),
    ("train_dataset_size", "TD", "int"),
    ("val_dataset_size", "VD", "int"),
    ("model_loss", "MLOSS", "float"),
    ("model_val_score", "MVSCORE", "float"),
    ("model_best_val_score", "MBVSCORE", "float"),
    ("num_factors", "F", "int"),
]


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


class LassoOneDTransitionRewardModel(OneDTransitionRewardModel):
    def __init__(
        self,
        model: Model,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[mbrl.types.ObsProcessFnType] = None,
        no_delta_list: Optional[List[int]] = None,
        num_elites: Optional[int] = None,
    ):
        super().__init__(
            model,
            target_is_delta,
            normalize,
            normalize_double_precision,
            learned_rewards,
            obs_process_fn,
            no_delta_list,
            num_elites,
        )

        assert hasattr(
            self.model, "lassonets"
        ), f"This Model Wrapper only works for Model with lassonets, model is {model._get_name()}"

    def fix_model_sparsity(self, factors, reinit: bool = False) -> None:

        self.model.set_factors(factors)

        if reinit:
            model = FactoredSimple(
                in_size=self.model.in_size,
                out_size=self.model.out_size,
                factors=factors,
                device=self.model.device,
                num_layers=self.model.num_layers,
                hid_size=self.model.hid_size,
                activation_fn_cfg=self.model.activation_fn_cfg,
            )
            self.model = model

        else:
            with torch.no_grad():
                for output, factor in enumerate(factors):
                    theta = self.model.lassonets[output].skip.weight.data.squeeze()
                    theta[~factor] = 0.0
                    weights0 = (
                        self.model.lassonets[output].layers[0].weight.data.squeeze()
                    )
                    mask = torch.zeros_like(weights0)
                    for input_ in factor:
                        mask[:, input_] = torch.ones_like(weights0[:, input_])
                    mask = mask.bool()
                    self.model.lassonets[output].create_mask(mask)

    def lassonet_update(
        self,
        lassonet: LassoNet,
        which_output: int,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ):
        assert target is None
        model_in, target = self._process_batch(batch)
        assert target.ndim == 2
        target = target[:, which_output]
        target = target.unsqueeze(-1)
        return self.model.lassonet_update(
            lassonet, model_in, optimizer, target=target, lambda_=lambda_
        )

    def lassonet_eval_score(
        self,
        lassonet: LassoNet,
        which_output: int,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        assert target is None
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            assert target.ndim == 2
            target = target[:, which_output]
            target = target.unsqueeze(-1)
            return self.model.lassonet_eval_score(
                lassonet, model_in, target=target, lambda_=lambda_
            )


class LassoModelTrainer(ModelTrainer):

    _SPARSITY_LOG_GROUP_NAME = "lasso_sparsity"

    def __init__(
        self,
        model: LassoOneDTransitionRewardModel,
        env_bounds: Union[Tuple, List[Tuple]] = None,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
        lambda_start: float = 0.0,
        lambda_max: float = 50.0,
        lambda_step: float = 0.2,
        take_best_factors: Optional[int] = None,
        num_pretraining_epochs: int = 50,
        theta_tol: float = 0.01,
        reinit: bool = False,
    ):

        self.model = model
        assert hasattr(
            self.model.model, "lassonets"
        ), "This Model Trainer only works for Model with lassonets"

        self.lassonets = self.model.model.lassonets  # TODO wrapper
        self._train_iteration = 0
        self.lambda_start = lambda_start
        self.lambda_max = lambda_max
        self.lambda_step = lambda_step
        self.take_best_factors = take_best_factors
        self.num_pretraining_epochs = num_pretraining_epochs
        self.theta_tol = theta_tol
        self.reinit = reinit

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME, MODEL_LOG_FORMAT, color="blue", dump_frequency=1
            )
            self.logger.register_group(
                self._SPARSITY_LOG_GROUP_NAME,
                SPARSITY_LOG_FORMAT,
                color="blue",
                dump_frequency=1,
            )

        self.optimizer = []
        self.optim_lr = optim_lr
        self.weight_decay = weight_decay
        self.optim_eps = optim_eps

        for lassonet in self.lassonets:
            optimizer = optim.Adam(
                lassonet.parameters(),
                lr=optim_lr,
                weight_decay=weight_decay,
                eps=optim_eps,
            )
            self.optimizer.append(optimizer)

        if env_bounds:
            if isinstance(env_bounds, Tuple):
                assert len(env_bounds) == 2
                self.env_bounds = [env_bounds for i in range(self.model.in_size)]
            else:
                assert len(env_bounds[0]) == 2
                self.env_bounds = env_bounds
        else:
            self.env_bounds = None

    def _reinit(self):

        # TODO: Write it better
        assert isinstance(self.model.model, FactoredSimple)
        self.sub_models = self.model.model.models
        self.lassonets = None

        self._train_iteration = 0

        self.optimizer = []
        for model in self.sub_models:
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.optim_lr,
                weight_decay=self.weight_decay,
                eps=self.optim_eps,
            )
            self.optimizer.append(optimizer)

    def find_sparse_model(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        callback_sparsity: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ):

        all_factors = []
        for which_output, lassonet in enumerate(self.lassonets):
            print("LASSONET", which_output)
            factors, best_lambda, best_train_loss, best_eval_loss, best_thetas = self._train_and_find_sparsity_lassonet(
                dataset_train,
                lassonet,
                which_output,
                dataset_val,
                callback_sparsity,
                evaluate,
                silent,
            )
            # factors = self._choose_factors(factors, best_thetas)
            all_factors.append(factors)
            # TODO: if ever have a model that learn dynamics and sparsity simultaneously, might be useful to keep
            # the best lambda for each submodel

        # TODO: make it cleaner
        print("all_factors", all_factors)
        callback_sparsity(None, None, None, all_factors)

        self.model.fix_model_sparsity(all_factors, reinit=self.reinit)
        if self.reinit:
            self._reinit()

        return all_factors

    def _choose_factors(self, factors, thetas):
        """
        Function called after "training_and_eval_sparsity", based on the informations
        given, will choose how to factor the model
        """

        # TODO: Function not robust or good at all
        if self.take_best_factors is not None:
            return factors
        else:
            theta_factors = [
                i for i, theta in enumerate(thetas) if theta > 2 * self.theta_tol
            ]
            return theta_factors

    def _train_and_find_sparsity_lassonet(
        self,
        dataset_train: TransitionIterator,
        lassonet: LassoNetAdapted,
        which_output: int,
        dataset_val: Optional[TransitionIterator] = None,
        callback_sparsity: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ) -> Tuple:

        current_lambda = self.lambda_start
        lambda_iters = ceil((self.lambda_max - self.lambda_start) / self.lambda_step)

        all_train_loss = []
        all_eval = []
        all_thetas = np.empty((lassonet.in_size, lambda_iters))
        lambda_count = 0
        factors = np.arange(lassonet.in_size)
        num_epochs = self.num_pretraining_epochs

        while lambda_count < lambda_iters:
            # Same as mother class ModelTrainer train function but gives lambda to update
            eval_dataset = dataset_train if dataset_val is None else dataset_val

            training_losses, val_scores = [], []
            epoch_iter = range(num_epochs)
            epochs_since_update = 0
            best_val_score = (
                self._evaluate_one_lassonet(
                    lassonet,
                    which_output=which_output,
                    dataset=eval_dataset,
                    lambda_=0.0,
                )
                if evaluate
                else None
            )
            # only enable tqdm if training for a single epoch,
            # otherwise it produces too much output
            disable_tqdm = silent or (num_epochs is None or num_epochs > 1)

            if lambda_count > 0:
                current_lambda += self.lambda_step

            thetas = np.empty((lassonet.in_size, num_epochs))
            for epoch in epoch_iter:
                batch_losses: List[float] = []
                for batch in tqdm.tqdm(dataset_train, disable=disable_tqdm):
                    loss, meta = self.model.lassonet_update(
                        lassonet,
                        which_output=which_output,
                        batch=batch,
                        optimizer=self.optimizer[which_output],
                        lambda_=current_lambda,
                    )
                    batch_losses.append(loss)
                total_avg_loss = np.mean(batch_losses).mean().item()
                training_losses.append(total_avg_loss)
                for j, weight in enumerate(lassonet.skip.weight.data.squeeze()):
                    thetas[j, epoch] = abs(weight)

                eval_score = None
                model_val_score = 0
                if evaluate:
                    eval_score = self._evaluate_one_lassonet(
                        lassonet,
                        which_output=which_output,
                        dataset=eval_dataset,
                        lambda_=0.0,
                    )
                    val_scores.append(eval_score.mean().item())
                    epochs_since_update += 1
                    model_val_score = eval_score.mean()

                if self.logger and not silent:
                    self.logger.log_data(  # TODO
                        self._SPARSITY_LOG_GROUP_NAME,
                        {
                            "lambda": current_lambda,
                            "lambda_iteration": lambda_count,
                            "epoch": epoch,
                            "train_dataset_size": dataset_train.num_stored,
                            "val_dataset_size": dataset_val.num_stored
                            if dataset_val is not None
                            else 0,
                            "model_loss": total_avg_loss,
                            "model_val_score": model_val_score,
                            "model_best_val_score": best_val_score.mean()
                            if best_val_score is not None
                            else 0,
                            "num_factors": len(factors),
                        },
                    )

                elif lassonet.selected_count() == 0:
                    warnings.warn(
                        "Features all disappeared, lambda might be already to high"
                    )
                    factors = []
                    break
            # #DEBUG
            # #Plot eval/training losses
            # fig, axs = plt.subplots(1, 2)
            # axs[0].plot(training_losses)
            # axs[0].set_xlabel("epoch iteration")
            # axs[0].set_ylabel("Training loss")
            # axs[1].plot(val_scores)
            # axs[1].set_xlabel("epoch iteration")
            # axs[1].set_ylabel("Eval loss")
            # plt.title(f"Lambda {current_lambda}")
            # plt.show()

            # #Theta (skip layer weights)
            # cols = self.model.model.in_size//2
            # fig, axs = plt.subplots(2, cols)
            # for weight_i, weights in enumerate(thetas):
            #     i = weight_i//cols
            #     j = weight_i%cols
            #     axs[i,j].plot(weights)
            #     axs[i,j].set_title(f"Theta {weight_i}, mean {np.mean(weights)}")
            #     axs[i,j].set_xlabel("epoch iteration")
            #     axs[i,j].set_ylabel("Theta")
            # plt.show()

            if training_losses[-1] - training_losses[0] > 0:
                print("Not training correctly anymore")
                num_epochs = 2  # TODO: Hard coded for now
                # Alternative solution: from this point just fix lambda

            best_epoch_train_loss = training_losses[-1]
            best_epoch_eval = val_scores[-1]
            best_epoch_thetas = np.mean(thetas[:, -10:], axis=1)

            all_train_loss.append(best_epoch_train_loss)
            all_eval.append(best_epoch_eval)
            for j, weight in enumerate(best_epoch_thetas):
                all_thetas[j, lambda_count] = weight
            lambda_count += 1

            factors = np.argwhere(
                np.array(best_epoch_thetas) > self.theta_tol
            ).squeeze()
            if self.take_best_factors is not None:
                if len(factors) <= self.take_best_factors:
                    print(f"Found the {self.take_best_factors} factors: {factors}")
                    break
            else:
                if len(factors) < lassonet.in_size:
                    print(f"First sparsity appeared")
                    factors = np.argwhere(
                        np.array(best_epoch_thetas) > 10 * self.theta_tol
                    ).squeeze()
                    # TODO: SHould be more precise or hyperparameter
                    break

        def create_callback_plots():
            plt.close("all")
            lambdas = np.arange(self.lambda_start, self.lambda_max, self.lambda_step)
            # Plot eval/training losses
            fig_loss, axs = plt.subplots(1, 2)
            axs[0].plot(lambdas[:lambda_count], all_train_loss)
            axs[0].set_xlabel("lambda")
            axs[0].set_ylabel("Training loss")
            axs[1].plot(lambdas[:lambda_count], all_eval)
            axs[1].set_xlabel("lambda")
            axs[1].set_ylabel("Eval loss")

            # Theta (skip layer weights)
            cols = lassonet.in_size // 2
            max_theta = np.max(all_thetas[:, :lambda_count]) + 0.1
            fig_theta, axs = plt.subplots(2, cols)
            for theta_i, theta in enumerate(all_thetas):
                i = theta_i // cols
                j = theta_i % cols
                axs[i, j].plot(lambdas[:lambda_count], abs(theta)[:lambda_count])
                axs[i, j].set_xlabel("lambda")
                axs[i, j].set_ylabel(f"Theta {theta_i}")
                axs[i, j].set_ylim(0, max_theta)

            return fig_loss, fig_theta

        if callback_sparsity:
            fig_loss, fig_theta = create_callback_plots()
            callback_sparsity(which_output, fig_loss, fig_theta)

        # Return the sparsity infos associated to the best lambda
        best_idx = np.argmin(all_train_loss)
        best_lambda = (
            self.lambda_start + best_idx * self.lambda_step
        )  # check if correct
        best_train_loss = all_train_loss[best_idx]
        best_eval_loss = all_eval[best_idx]
        best_thetas = all_thetas[:, best_idx]
        return factors, best_lambda, best_train_loss, best_eval_loss, best_thetas

    def _evaluate_one_lassonet(
        self,
        lassonet: LassoNet,
        which_output: int,
        dataset: TransitionIterator,
        batch_callback: Optional[Callable] = None,
        lambda_: Optional[float] = None,
    ) -> torch.Tensor:

        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        batch_scores_list = []
        for batch in dataset:
            batch_score, meta = self.model.lassonet_eval_score(
                lassonet, which_output, batch, lambda_=lambda_
            )
            batch_scores_list.append(batch_score)
            if batch_callback:
                batch_callback(batch_score.mean(), meta, "eval")
        try:
            batch_scores = torch.cat(
                batch_scores_list, dim=batch_scores_list[0].ndim - 2
            )
        except RuntimeError as e:
            print(
                f"There was an error calling ModelTrainer.evaluate(). "
                f"Note that model.eval_score() should be non-reduced. Error was: {e}"
            )
            raise e
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        mean_axis = 0 if batch_scores.ndim == 2 else (1, 2)
        batch_scores = batch_scores.mean(dim=mean_axis)

        return batch_scores

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        callback_sparsity: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ) -> Tuple[List[float], List[float]]:

        if self._train_iteration == 0:
            self.find_sparse_model(
                dataset_train, dataset_val, callback_sparsity, evaluate, silent
            )

        return super().train(
            dataset_train,
            dataset_val,
            num_epochs,
            patience,
            improvement_threshold,
            callback,
            batch_callback,
            evaluate,
            silent,
        )


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

        Model.__init__(self, device)  # If does not work try super().super().__init__()

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
        return F.mse_loss(pred_out, target, reduction="none").sum(-1).sum(), {}

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
            loss = F.mse_loss(pred_out, target, reduction="none").unsqueeze(0)
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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if lambda_ is None:
            warnings.warn("You did not give any lambda, default lambda = 0.")
            lambda_ = 0.0

        assert model_in.ndim == 2 and target.ndim == 2
        assert target.shape[-1] == len(self.lassonets)

        eval_scores = []
        metas = []
        # Compute the loss for each single output and its associated lassonet model
        for i, lassonet in enumerate(self.lassonets):
            sub_target = target[:, i].unsqueeze(-1)
            eval_score, meta = self.lassonet_eval_score(
                lassonet, model_in, sub_target, lambda_
            )
            eval_scores.append(eval_score)
            metas.append(meta)

        n_outputs = len(eval_scores)
        assert n_outputs == self.out_size

        return torch.cat(eval_scores, dim=-1), {}

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
        mode: str = "sum",
    ) -> Tuple[float, Dict[str, Any]]:

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
            all_ans.append(ans)
            all_meta.append(meta)

        n_outputs = len(all_ans)
        assert n_outputs == self.out_size
        if mode == "sum":
            return sum(all_ans), {}
        elif mode == "mean":
            return sum(all_ans) / n_outputs, {}
        elif mode == "separate":
            return all_ans, all_meta
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
            )
