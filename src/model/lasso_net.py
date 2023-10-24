from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import functools
import itertools
import tqdm
import warnings

import hydra
import numpy as np
import omegaconf
import torch
from torch import nn

import mbrl
from mbrl.util.replay_buffer import TransitionIterator, BootstrapIterator
from mbrl.models.model import Model
from mbrl.models.model_trainer import ModelTrainer
from mbrl.models.one_dim_tr_model import OneDTransitionRewardModel
from mbrl.util.logger import Logger
from lassonet import LassoNet


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

        super().__init__()

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
        self.layers = nn.Sequential(*hidden_layers)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.skip = nn.Linear(in_size, out_size, bias=False)


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
        optimizer = self.optimizer

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
            if (
                hidden_layers.selected_count() == 0 or repeat > 3
            ):  # TODO: loop is not great at all !!!!
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
