from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import functools
import itertools
import tqdm
import warnings

import hydra
import numpy as np
import omegaconf
import torch
from torch import optim
from torch import nn

import mbrl
from mbrl.util.replay_buffer import TransitionIterator, BootstrapIterator
from mbrl.models.model import Model
from mbrl.models.model_trainer import ModelTrainer, MODEL_LOG_FORMAT
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
        current_layer = x
        result = self.skip(x)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = self.action_layer(current_layer)
        return result + current_layer


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

    def update(
        self,
        batch: mbrl.types.TransitionBatch,
        optimizers: List[torch.optim.Optimizer],
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        assert target is None
        model_in, target = self._process_batch(batch)
        return self.model.update(model_in, optimizers, target=target, lambda_=lambda_)

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
    def __init__(
        self,
        model: LassoOneDTransitionRewardModel,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
        lambda_seq: Optional[float] = None,
        lambda_start: Optional[float] = None,
        path_multiplier: float = 1.02,
    ):

        self.model = model
        self._train_iteration = 0

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME, MODEL_LOG_FORMAT, color="blue", dump_frequency=1
            )

        self.optimizers = []
        for lassonet in self.model.model.lassonets:
            optimizer = optim.Adam(
                lassonet.parameters(),
                lr=optim_lr,
                weight_decay=weight_decay,
                eps=optim_eps,
            )
            self.optimizers.append(optimizer)

        self.lambda_seq = lambda_seq
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier

        assert hasattr(
            self.model.model, "lassonets"
        ), "This Model Trainer only works for Model with lassonets"

    def evaluate_one_lassonet(
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

    def evaluate(
        self,
        dataset: TransitionIterator,
        batch_callback: Optional[Callable] = None,
        lambda_: Optional[float] = None,
        mode: str = "mean",
    ) -> torch.Tensor:

        all_batch_scores = []
        for i, lassonet in enumerate(self.model.model.lassonets):
            batch_scores = self.evaluate_one_lassonet(
                lassonet,
                which_output=i,
                dataset=dataset,
                batch_callback=batch_callback,
                lambda_=lambda_,
            )
            all_batch_scores.append(batch_scores)

        if mode == "mean":
            n_scores = len(all_batch_scores)
            assert n_scores == self.model.model.out_size
            return torch.stack(all_batch_scores, dim=0).sum(dim=0) / n_scores
        elif mode == "separate":
            return all_batch_scores
        else:
            raise ValueError(
                f"There is no {mode} mode for the SimpleLasso eval_score method"
            )

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        callback_sparsity: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
        lambda_seq: List = None,
        lambda_max=float("inf"),
    ) -> Tuple[List[float], List[float]]:
        """
        TODO: Works well if each output has the same sparsity level. Otherwise, need to adapt the function

        TODO: Review, polish and finish the function
        TODO: Goal of the Wrapper is not to have the self.model.model, add functions in wrapper !!
        Highly based on LassoNetRegressor Path function
        """

        print("Is is X_train[1]???", dataset_train.num_stored)

        # Check that
        lassonet_ref = self.model.model.lassonets[0]
        optimizer_ref = self.optimizers[0]

        # hidden_layers = self.model.model.hidden_layers

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
                    lassonet_ref.lambda_start(M=self.model.model.M)
                    / optimizer_ref.param_groups[0]["lr"]
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
        for current_lambda in [
            0.1
        ]:  # itertools.chain([lambda_start], lambda_seq): TODO
            if (
                lassonet_ref.selected_count() == 0  # or repeat > 3
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
                self.evaluate(dataset=eval_dataset, lambda_=current_lambda)
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
                        batch, self.optimizers, lambda_=current_lambda
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

            if is_dense and lassonet_ref.selected_count() < dataset_train.num_stored:
                is_dense = False
            if current_lambda / lambda_start < 2:
                warnings.warn(
                    f"lambda_start={lambda_start:.3f} "
                    f"{'(selected automatically) ' * (self.lambda_start == 'auto')}"
                    "might be too large.\n"
                    f"Features start to disappear at {current_lambda=:.3f}."
                )

            if callback_sparsity:
                model_features_counts = {}
                for i, lassonet in enumerate(self.model.model.lassonets):
                    name = f"lassonet_{i}"
                    model_features_counts[name] = lassonet.selected_count()
                callback_sparsity(model_features_counts, current_lambda)

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
