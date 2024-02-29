# Overriden of model trainer from mbrl library (with small modifs) and and daughter class
import functools
import itertools
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import tqdm
import torch
from torch import optim
from torcheval.metrics.functional import r2_score
from matplotlib import pyplot as plt

import mbrl
from mbrl.models.model_trainer import (
    Model,
    ModelTrainer,
    MODEL_LOG_FORMAT,
    Logger,
    BootstrapIterator,
    TransitionIterator,
)

from lassonet import LassoNet

# from src.model.simple import FactoredSimple
from src.model.lasso_net import LassoNetAdapted
from src.model.simple import FactoredSimple
from src.model.model_mixture import MixtureModel

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

ADD_MODEL_LOG_FORMAT = [("train_R2", "TR2", "float"), ("eval_R2", "VR2", "float")]

MODEL_LOG_FORMAT += ADD_MODEL_LOG_FORMAT


class ModelTrainerOverriden(ModelTrainer):
    """Trainer for dynamics models. Override of ModelTrainer from mbrl

    Args:
        model (:class:`mbrl.models.Model`): a model to train.
        optim_lr (float): the learning rate for the optimizer (using Adam).
        weight_decay (float): the weight decay to use.
        logger (:class:`mbrl.util.Logger`, optional): the logger to use.
    """

    def __init__(
        self,
        model: Model,
        optim_lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
    ):
        super().__init__(model, optim_lr, weight_decay, optim_eps, logger)

    def train(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        improvement_threshold: float = 0.01,
        callback: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        split_callback: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
        debug: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Trains the model for some number of epochs.

        This method iterates over the stored train dataset, one batch of transitions at a time,
        updates the model.

        If a validation dataset is provided in the constructor, this method will also evaluate
        the model over the validation data once per training epoch. The method will keep track
        of the weights with the best validation score, and after training the weights of the
        model will be set to the best weights. If no validation dataset is provided, the method
        will keep the model with the best loss over training data.

        Args:
            dataset_train (:class:`mbrl.util.TransitionIterator`): the iterator to
                use for the training data.
            dataset_val (:class:`mbrl.util.TransitionIterator`, optional):
                an iterator to use for the validation data.
            num_epochs (int, optional): if provided, the maximum number of epochs to train for.
                Default is ``None``, which indicates there is no limit.
            patience (int, optional): if provided, the patience to use for training. That is,
                training will stop after ``patience`` number of epochs without improvement.
                Ignored if ``evaluate=False`.
            improvement_threshold (float): The threshold in relative decrease of the evaluation
                score at which the model is seen as having improved.
                Ignored if ``evaluate=False`.
            callback (callable, optional): if provided, this function will be called after
                every training epoch with the following positional arguments::

                    - the model that's being trained
                    - total number of calls made to ``trainer.train()``
                    - current epoch
                    - training loss
                    - validation score (for ensembles, factored per member)
                    - best validation score so far

            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.update()`` (during training),
                and ``model.eval_score()`` (during evaluation). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is one of ``"train"`` or ``"eval"``, indicating if the callback
                was called during training or evaluation.

            evaluate (bool, optional): if ``True``, the trainer will use ``model.eval_score()``
                to keep track of the best model. If ``False`` the model will not compute
                an evaluation score, and simply train for some number of epochs. Defaults to
                ``True``.

            silent (bool): if ``True`` logging and progress bar are deactivated. Defaults
                to ``False``.

        Returns:
            (tuple of two list(float)): the history of training losses and validation losses.

        """
        eval_dataset = dataset_train if dataset_val is None else dataset_val

        training_losses, val_scores, train_r2_scores, eval_r2_scores = [], [], [], []
        best_weights: Optional[Dict] = None
        epoch_iter = range(num_epochs) if num_epochs else itertools.count()
        epochs_since_update = 0
        best_val_score, _ = self.evaluate(eval_dataset) if evaluate else None
        # only enable tqdm if training for a single epoch,
        # otherwise it produces too much output
        disable_tqdm = silent or (num_epochs is None or num_epochs > 1)
        if debug:
            from time import time

        for epoch in epoch_iter:
            if debug:
                start = time()
            if batch_callback:
                batch_callback_epoch = functools.partial(batch_callback, epoch)
            else:
                batch_callback_epoch = None
            batch_losses: List[float] = []
            batch_r2_scores: List[float] = []
            for batch in tqdm.tqdm(dataset_train, disable=disable_tqdm):
                loss, meta = self.model.update(batch, self.optimizer)
                batch_losses.append(loss)
                if "outputs" in meta and "targets" in meta:
                    batch_r2_scores.append(r2_score(meta["outputs"], meta["targets"]))
                if batch_callback_epoch:
                    batch_callback_epoch(loss, meta, "train")
            total_avg_loss = np.mean(batch_losses).mean().item()
            training_losses.append(total_avg_loss)
            if batch_r2_scores:
                train_r2_scores.append(np.mean(batch_r2_scores))

            eval_score = None
            model_val_score = 0
            if evaluate:
                if split_callback:
                    dyn_eval_score, rew_eval_score, dyn_r2, rew_r2 = self.evaluate(
                        eval_dataset, batch_callback=batch_callback_epoch, split=True
                    )
                    eval_score = torch.mean(
                        torch.cat([dyn_eval_score, rew_eval_score], axis=-1), dim=-1
                    )
                    r2 = np.mean([dyn_r2, rew_r2])
                else:
                    eval_score, r2 = self.evaluate(
                        eval_dataset, batch_callback=batch_callback_epoch
                    )
                val_scores.append(eval_score.mean().item())
                if r2:
                    eval_r2_scores.append(r2)

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

            if debug:
                end = time()
                print(f"Training epoch duration: {round(end-start, 2)}s")
                debug_log = (
                    f"Epoch: {epoch} Train loss {total_avg_loss:.3f}, "
                    f"Test loss {model_val_score:.3f} "
                )
                if train_r2_scores:
                    debug_log += f"Train R2 {train_r2_scores[-1]:.3f}, "
                if eval_r2_scores:
                    debug_log += f"Test R2 {eval_r2_scores[-1]:.3f} "
                print(debug_log)

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
                        "train_R2": train_r2_scores[-1] if train_r2_scores else None,
                        "eval_R2": eval_r2_scores[-1] if eval_r2_scores else None,
                    },
                )
            if callback:
                callback(
                    self.model,
                    self._train_iteration,
                    epoch,
                    total_avg_loss,
                    model_val_score,
                    best_val_score.mean(),
                    train_r2_scores[-1] if train_r2_scores else None,
                    eval_r2_scores[-1] if eval_r2_scores else None,
                )
            if split_callback:
                split_callback(
                    self._train_iteration,
                    epoch,
                    dyn_eval_score,
                    rew_eval_score,
                    dyn_r2,
                    rew_r2,
                )

            if patience and epochs_since_update >= patience:
                break

        # saving the best models:
        if evaluate:
            self._maybe_set_best_weights_and_elite(best_weights, best_val_score)

        self._train_iteration += 1
        return training_losses, val_scores, train_r2_scores, eval_r2_scores

    def evaluate(
        self,
        dataset: TransitionIterator,
        batch_callback: Optional[Callable] = None,
        split: bool = False,
    ) -> torch.Tensor:
        """Evaluates the model on the validation dataset.

        Iterates over the dataset, one batch at a time, and calls
        :meth:`mbrl.models.Model.eval_score` to compute the model score
        over the batch. The method returns the average score over the whole dataset.

        Args:
            dataset (bool): the transition iterator to use.
            batch_callback (callable, optional): if provided, this function will be called
                for every batch with the output of ``model.eval_score()`` (the score will
                be passed as a float, reduced using mean()). It will be called
                with four arguments ``(epoch_index, loss/score, meta, mode)``, where
                ``mode`` is the string ``"eval"``.

        Returns:
            (tensor): The average score of the model over the dataset (and for ensembles, per
                ensemble member).
        """
        if isinstance(dataset, BootstrapIterator):
            dataset.toggle_bootstrap()

        batch_scores_list = []
        if split:
            dyn_batch_r2_scores = []
            rew_batch_r2_scores = []
        else:
            batch_r2_scores = []
        for batch in dataset:
            batch_score, meta = self.model.eval_score(batch)
            batch_scores_list.append(batch_score)
            if "outputs" in meta and "targets" in meta:
                if split:
                    dyn_batch_r2_scores.append(
                        r2_score(meta["outputs"][..., :-1], meta["targets"][..., :-1])
                    )
                    rew_batch_r2_scores.append(
                        r2_score(meta["outputs"][..., -1], meta["targets"][..., -1])
                    )
                else:
                    batch_r2_scores.append(r2_score(meta["outputs"], meta["targets"]))
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
        if split:
            dyn_batch_scores = batch_scores[..., :-1]
            rew_batch_scores = batch_scores[..., -1][..., None]
            dyn_batch_scores = dyn_batch_scores.mean(dim=mean_axis)
            rew_batch_scores = rew_batch_scores.mean(dim=mean_axis)
            dyn_r2 = np.mean(dyn_batch_r2_scores) if dyn_batch_r2_scores else None
            rew_r2 = np.mean(rew_batch_r2_scores) if rew_batch_r2_scores else None
            return dyn_batch_scores, rew_batch_scores, dyn_r2, rew_r2

        batch_scores = batch_scores.mean(dim=mean_axis)
        r2 = np.mean(batch_r2_scores) if batch_r2_scores else None
        return batch_scores, r2


class MixtureModelsTrainer(ModelTrainerOverriden):
    def __init__(
        self,
        model: Model,
        rew_optim_lr: float = 1e-1,
        dyn_optim_lr: float = 1e-4,
        rew_weight_decay: float = 0.0,
        dyn_weight_decay: float = 1e-5,
        rew_optim_eps: float = 1e-8,
        dyn_optim_eps: float = 1e-8,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self._train_iteration = 0

        self.logger = logger
        if self.logger:
            self.logger.register_group(
                self._LOG_GROUP_NAME, MODEL_LOG_FORMAT, color="blue", dump_frequency=1
            )

        assert isinstance(
            self.model.model, MixtureModel
        ), f"But you are using model {self.model.__class__.__name__}"

        self.optimizer = [
            optim.Adam(
                self.model.model.dyn_model.parameters(),
                lr=dyn_optim_lr,
                weight_decay=dyn_weight_decay,
                eps=dyn_optim_eps,
            )
        ]
        self.optimizer.append(
            optim.Adam(
                self.model.model.reward_model.parameters(),
                lr=rew_optim_lr,
                weight_decay=rew_weight_decay,
                eps=rew_optim_eps,
            )
        )


class LassoModelTrainer(ModelTrainer):
    _SPARSITY_LOG_GROUP_NAME = "lasso_sparsity"

    def __init__(
        self,
        model: Model,
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
        self.unwrapped_model = model.model
        assert hasattr(
            self.unwrapped_model, "lassonets"
        ), "This Model Trainer only works for Model with lassonets"

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

        for lassonet in self.unwrapped_model.lassonets:
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
        assert isinstance(self.unwrapped_model, FactoredSimple)
        self.sub_models = self.unwrapped_model.models
        self.unwrapped_model.lassonets = None

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

    def fix_model_sparsity(self, factors, reinit: bool = False) -> None:
        self.unwrapped_model.set_factors(factors)
        if reinit:
            model = FactoredSimple(
                in_size=self.unwrapped_model.in_size,
                out_size=self.unwrapped_model.out_size,
                factors=factors,
                device=self.unwrapped_model.device,
                num_layers=self.unwrapped_model.num_layers,
                hid_size=self.unwrapped_model.hid_size,
                activation_fn_cfg=self.unwrapped_model.activation_fn_cfg,
            )
            self.unwrapped_model = model
        else:
            with torch.no_grad():
                for output, factor in enumerate(factors):
                    theta = self.unwrapped_model.lassonets[
                        output
                    ].skip.weight.data.squeeze()
                    theta[~factor] = 0.0
                    weights0 = (
                        self.unwrapped_model.lassonets[output]
                        .layers[0]
                        .weight.data.squeeze()
                    )
                    mask = torch.zeros_like(weights0)
                    for input_ in factor:
                        mask[:, input_] = torch.ones_like(weights0[:, input_])
                    mask = mask.bool()
                    self.unwrapped_model.lassonets[output].create_mask(mask)

    def lassonet_update(
        self,
        lassonet: LassoNet,
        which_output: int,
        batch: mbrl.types.TransitionBatch,
        optimizer: torch.optim.Optimizer,
        target: Optional[torch.Tensor] = None,
        lambda_: float = None,
    ):
        """
        Bypass the classic model wrapper update to allow the update
        of one lassonet at a time instead

        :param lassonet: lassonet to update
        :param which_output: which factor (output) is related to this lassonet
        :param lambda_: sparsity hyperparameter, defaults to None
        :return: loss of the updated lassonet
        """
        assert target is None
        model_in, target = self.model._process_batch(batch)
        assert target.ndim == 2
        target = target[:, which_output]
        target = target.unsqueeze(-1)
        return self.unwrapped_model.lassonet_update(
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
        """
        Bypass the classic model wrapper update to allow the score evaulation
        of one lassonet at a time instead

        :param lassonet: lassonet to update
        :param which_output: which factor (output) is related to this lassonet
        :param lambda_: sparsity hyperparameter, defaults to None
        :return: eval loss of the updated lassonet
        """

        assert target is None
        with torch.no_grad():
            model_in, target = self.model._process_batch(batch)
            assert target.ndim == 2
            target = target[:, which_output]
            target = target.unsqueeze(-1)
            return self.unwrapped_model.lassonet_eval_score(
                lassonet, model_in, target=target, lambda_=lambda_
            )

    def find_sparse_model(
        self,
        dataset_train: TransitionIterator,
        dataset_val: Optional[TransitionIterator] = None,
        callback_sparsity: Optional[Callable] = None,
        evaluate: bool = True,
        silent: bool = False,
    ):
        all_factors = []
        for which_output, lassonet in enumerate(self.unwrapped_model.lassonets):
            (
                factors,
                best_lambda,
                best_train_loss,
                best_eval_loss,
                best_thetas,
            ) = self._train_and_find_sparsity_lassonet(
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

        self.fix_model_sparsity(all_factors, reinit=self.reinit)
        if self.reinit:
            self._reinit()

        return all_factors

    def _choose_factors(self, factors, thetas):
        """
        Function called after "training_and_eval_sparsity", based on the informations
        given, will choose how to factor the model
        """

        # TODO: Function not robust or good at all
        # No theory behind, should be investigated
        # NOT USED FOR NOW
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
                    loss, meta = self.lassonet_update(
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
                    self.logger.log_data(
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

            if training_losses[-1] - training_losses[0] > 0:
                print("Not training correctly anymore")
                num_epochs = 2  # TODO: Hard coded for now
                # TODO: Alternative solution: from this point just fix lambda

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
                    # TODO: Should be more precise or give a hyperparameter
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
        )  # TODO: check if correct
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
            batch_score, meta = self.lassonet_eval_score(
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
        debug: bool = False, #Unused for now anyway
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
