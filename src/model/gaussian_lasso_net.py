import functools
import itertools
import sys
import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import omegaconf
import torch
from torch.nn import functional as F
from torch import optim as optim

import mbrl
from mbrl.types import ModelInput
from mbrl.models.gaussian_mlp import GaussianMLP
from mbrl.util.replay_buffer import TransitionIterator, BootstrapIterator
from mbrl.models.model import Model
from mbrl.models.model_trainer import ModelTrainer
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.util.logger import Logger
from lassonet.model import LassoNet

# from lassonet import LassoNetRegressor


class LassoNetGaussianMLP(GaussianMLP):
    """
    Simple factored GaussianMLP model
    Should be generalized as we move to more complex environments

    Instead of processing the whole state-action input tensor of dimension [state_space+state_action],
    It processes each factored state-action input of dimension [factored_in_size] such that the output
    is simply the concatenation of all the outputs of the factored state-action inputs.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        gamma: float = 0.0,
        gamma_skip: float = 0.0,
        M: float = 10.0,
    ):

        super().__init__(
            in_size,
            out_size,
            device,
            num_layers,
            ensemble_size,
            hid_size,
            deterministic,
            propagation_method,
            learn_logvar_bounds,
            activation_fn_cfg,
        )

        self.hidden_dims = tuple([hid_size for layer in range(num_layers)])

        self.hidden_layers = LassoNet(in_size, *self.hidden_dims)
        self.gamma = gamma
        self.gamma_skip = gamma_skip
        self.M = M

    def eval_score(  # type: ignore
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        if lambda_ is None:
            warnings.warn("You are not evaluating using the lasso solution")
            return super().eval_score(model_in, target)

        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            loss = F.mse_loss(pred_mean, target, reduction="none")
            return (
                (
                    loss  # .item()
                    + lambda_ * self.hidden_layers.l1_regularization_skip().item()
                    + self.gamma * self.hidden_layers.l2_regularization().item()
                    + self.gamma_skip
                    * self.hidden_layers.l2_regularization_skip().item()
                ),
                {},
            )

    def update(
        self,
        model_in: ModelInput,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[float, Dict[str, Any]]:

        # TODO: Necessary ??
        if lambda_ is None:
            warnings.warn("You are not updating the model using Lasso")
            return super().update(model_in, optimizer, target)

        self.train()

        def closure():
            optimizer.zero_grad()
            loss, meta = self.loss(model_in, target)
            ans = (
                # TODO: Might not be a good idea full_loss + hiddenlayers reg
                loss
                + self.gamma * self.hidden_layers.l2_regularization()
                + self.gamma_skip * self.hidden_layers.l2_regularization_skip()
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

        # TODO: Useful ??
        self.hidden_layers.prox(
            lambda_=lambda_ * optimizer.param_groups[0]["lr"], M=self.M
        )

        return ans.item(), meta


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

    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        assert target is None
        model_in, target = self._process_batch(batch)
        return self.model.update(model_in, optimizer, target=target, lambda_=lambda_)

    def eval_score(
        self,
        batch: mbrl.types.TransitionBatch,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        assert target is None
        with torch.no_grad():
            model_in, target = self._process_batch(batch)
            return self.model.eval_score(model_in, target=target, lambda_=lambda_)


class LassoModelTrainer(ModelTrainer):
    def __init__(
        self,
        model: Model,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
        lambda_seq: Optional[float] = None,
        lambda_start: Optional[float] = None,
        path_multiplier: float = 1.02,
    ):

        super().__init__(model, optim_lr, weight_decay, optim_eps, logger)
        self.lambda_seq = lambda_seq
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier

    def evaluate(
        self,
        dataset: TransitionIterator,
        batch_callback: Optional[Callable] = None,
        lambda_: Optional[float] = None,
    ) -> torch.Tensor:

        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        batch_scores_list = []
        for batch in dataset:
            batch_score, meta = self.model.eval_score(batch, lambda_=lambda_)
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

        mean_axis = 1 if batch_scores.ndim == 2 else (1, 2)
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
        batch_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
        lambda_seq: List = None,
        lambda_max=float("inf"),
    ) -> Tuple[List[float], List[float]]:
        """
        TODO: Review, polish and finish the function
        Highly based on LassoNetRegressor Path function

        :param dataset_train: _description_
        :param dataset_val: _description_, defaults to None
        :param num_epochs: _description_, defaults to None
        :param patience: _description_, defaults to None
        :param improvement_threshold: _description_, defaults to 0.01
        :param callback: _description_, defaults to None
        :param batch_callback: _description_, defaults to None
        :param evaluate: _description_, defaults to True
        :param silent: _description_, defaults to False
        :param lambda_seq: _description_, defaults to None
        :param lambda_max: _description_, defaults to float("inf")
        :return: _description_
        :yield: _description_
        """

        print("Is is X_train[1]???", dataset_train.num_stored)

        # Check that
        optimizer = self.optimizer  # self.optim_path(self.model.parameters())

        hidden_layers = self.model.model.hidden_layers

        # build lambda_seq
        if lambda_seq is not None:
            pass
        elif self.lambda_seq is not None:
            lambda_seq = self.lambda_seq
        else:

            def _lambda_seq(start):
                while start <= lambda_max:
                    yield start
                    start *= self.path_multiplier

            if self.lambda_start == "auto" or self.lambda_start is None:
                # divide by 10 for initial training
                self.lambda_start_ = (
                    hidden_layers.lambda_start(M=self.model.model.M)
                    / optimizer.param_groups[0]["lr"]
                    / 10
                )
                lambda_seq = _lambda_seq(self.lambda_start_)
            else:
                lambda_seq = _lambda_seq(self.lambda_start)

        # extract first value of lambda_seq
        lambda_seq = iter(lambda_seq)
        lambda_start = next(lambda_seq)

        is_dense = True
        repeat = 0
        for current_lambda in itertools.chain([lambda_start], lambda_seq):
            if hidden_layers.selected_count() == 0 or repeat > 3:
                break
            repeat += 1

            # Same as mother class ModelTrainer train function but gives lambda to update
            eval_dataset = dataset_train if dataset_val is None else dataset_val

            training_losses, val_scores = [], []
            best_weights: Optional[Dict] = None
            epoch_iter = range(num_epochs) if num_epochs else itertools.count()
            epochs_since_update = 0
            best_val_score = (
                self.evaluate(eval_dataset, lambda_=current_lambda)
                if evaluate
                else None
            )
            # only enable tqdm if training for a single epoch,
            # otherwise it produces too much output
            disable_tqdm = silent or (num_epochs is None or num_epochs > 1)

            for epoch in epoch_iter:
                if batch_callback:
                    batch_callback_epoch = functools.partial(batch_callback, epoch)
                else:
                    batch_callback_epoch = None
                batch_losses: List[float] = []
                for batch in tqdm.tqdm(dataset_train, disable=disable_tqdm):
                    loss, meta = self.model.update(
                        batch, self.optimizer, lambda_=current_lambda
                    )
                    batch_losses.append(loss)
                    if batch_callback_epoch:
                        batch_callback_epoch(loss, meta, "train")
                total_avg_loss = np.mean(batch_losses).mean().item()
                training_losses.append(total_avg_loss)

                eval_score = None
                model_val_score = 0
                if evaluate:
                    eval_score = self.evaluate(
                        eval_dataset,
                        batch_callback=batch_callback_epoch,
                        lambda_=current_lambda,
                    )
                    val_scores.append(eval_score.mean().item())

                    maybe_best_weights = self.maybe_get_best_weights(
                        best_val_score, eval_score, improvement_threshold
                    )
                    if maybe_best_weights:
                        best_val_score = torch.minimum(best_val_score, eval_score)
                        best_weights = maybe_best_weights
                        epochs_since_update = 0
                    else:
                        epochs_since_update += 1
                    model_val_score = eval_score.mean()

                if self.logger and not silent:
                    self.logger.log_data(
                        self._LOG_GROUP_NAME,
                        {
                            "iteration": self._train_iteration,
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
                        },
                    )
                if callback:
                    callback(
                        self.model,
                        self._train_iteration,
                        epoch,
                        total_avg_loss,
                        eval_score,
                        best_val_score,
                        # TODO: current_lambda,
                    )

                if patience and epochs_since_update >= patience:
                    break

            # saving the best models:
            if evaluate:
                self._maybe_set_best_weights_and_elite(best_weights, best_val_score)

            self._train_iteration += 1

            if is_dense and hidden_layers.selected_count() < dataset_train.num_stored:
                is_dense = False
            if current_lambda / lambda_start < 2:
                warnings.warn(
                    f"lambda_start={lambda_start:.3f} "
                    f"{'(selected automatically) ' * (self.lambda_start == 'auto')}"
                    "might be too large.\n"
                    f"Features start to disappear at {current_lambda=:.3f}."
                )

        return training_losses, val_scores

        # TODO: Add callbacks, verbose, hist, features_importance ????
        #     if self.verbose > 1:
        #         print(
        #             f"Lambda = {current_lambda:.2e}, "
        #             f"selected {self.model.selected_count()} features "
        #         )
        #         last.log()

        # self.feature_importances_ = self._compute_feature_importances(hist)
        # """When does each feature disappear on the path?"""
