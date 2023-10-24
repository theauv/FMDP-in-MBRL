import git
import logging
import pathlib
from typing import Dict, Union, Tuple, Optional, Callable
import yaml
import warnings

import gymnasium as gym
import hydra
import omegaconf
import numpy as np
import torch


import mbrl
from mbrl.util.replay_buffer import ReplayBuffer
from mbrl.models.gaussian_mlp import GaussianMLP
from lassonet.model import LassoNet

from src.model.lasso_net import LassoModelTrainer


def getBack(var_grad_fn):
    """
    Observe the backward graph functions of a tensor
    :param var_grad_fn: loss.grad_fn
    """
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def get_weights_model(model: torch.nn.Module, verbose: bool = False):

    with torch.no_grad():

        biases = []
        weights = []
        deterministic = True
        for i, (name, param) in enumerate(model.named_parameters()):

            if verbose:
                print(name)
                print(param.shape)

            names = name.split(".")

            if "skip" in names:
                skip = torch.squeeze(param).numpy()
            elif "mean_and_logvar" in names:
                if "bias" in names:
                    mean_and_logvar_biases = torch.squeeze(param).numpy()
                elif "weight" in names:
                    mean_and_logvar_weights = torch.squeeze(param).numpy()
                else:
                    raise ValueError(f"Weird mean_and_logvar named {name}")
            elif "min_logvar" in names:
                deterministic = False
                min_logvar = torch.squeeze(param).numpy()
            elif "max_logvar" in names:
                max_logvar = torch.squeeze(param).numpy()
            elif "hidden_layers" in names or "layers" in names:
                if "bias" in names:
                    bias = torch.squeeze(param).numpy()
                    biases.append(bias)
                elif "weight" in names:
                    weight = torch.squeeze(param).numpy()
                    weights.append(weight)
                else:
                    raise ValueError(f"Weird hidden layer named {name}")

        # if isinstance(model, LassoNetGaussianMLP):
        #     all_weights = {
        #         "hidden_weights": weights,
        #         "hidden_biases": biases,
        #         "mean_and_logvar_weights": mean_and_logvar_weights,
        #         "mean_and_logvar_biases": mean_and_logvar_biases,
        #         "skip": skip,
        #     }
        #     if not deterministic:
        #         all_weights["min_logvar"] = min_logvar
        #         all_weights["max_logvar"] = max_logvar
        if isinstance(model, GaussianMLP):
            all_weights = {
                "hidden_weights": weights,
                "hidden_biases": biases,
                "mean_and_logvar_weights": mean_and_logvar_weights,
                "mean_and_logvar_biases": mean_and_logvar_biases,
            }
            if not deterministic:
                all_weights["min_logvar"] = min_logvar
                all_weights["max_logvar"] = max_logvar
        elif isinstance(model, LassoNet):
            all_weights = {
                "hidden_weights": weights,
                "hidden_biases": biases,
                "skip": skip,
            }
        else:
            all_weights = {"hidden_weights": weights, "hidden_biases": biases}
            warnings.warn("This function might not support this model architecture")
        return all_weights


# TODO: Might put the following functions somewhere else


def train_model_and_save_model_and_data_overriden(
    model: mbrl.models.Model,
    model_trainer: mbrl.models.ModelTrainer,
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    work_dir: Optional[Union[str, pathlib.Path]] = None,
    callback: Optional[Callable] = None,
    callback_sparsity: Optional[Callable] = None,
):
    """
    Overwrite of function train_model_and_save_model_and_data in mbrl.util.common
    """

    dataset_train, dataset_val = mbrl.util.common.get_basic_buffer_iterators(
        replay_buffer,
        cfg.model_batch_size,
        cfg.validation_ratio,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=cfg.get("bootstrap_permutes", False),
    )
    if hasattr(model, "update_normalizer"):
        model.update_normalizer(replay_buffer.get_all())
    if isinstance(model_trainer, LassoModelTrainer):
        model_trainer.train(
            dataset_train,
            dataset_val=dataset_val,
            num_epochs=cfg.get("num_epochs_train_model", None),
            patience=cfg.get("patience", 1),
            improvement_threshold=cfg.get("improvement_threshold", 0.01),
            callback=callback,
            callback_sparsity=callback_sparsity,
        )
    else:
        model_trainer.train(
            dataset_train,
            dataset_val=dataset_val,
            num_epochs=cfg.get("num_epochs_train_model", None),
            patience=cfg.get("patience", 1),
            improvement_threshold=cfg.get("improvement_threshold", 0.01),
            callback=callback,
        )
    if work_dir is not None:
        model.save(str(work_dir))
        replay_buffer.save(work_dir)


def step_env_and_add_to_buffer_callback(
    env: gym.Env,
    obs: np.ndarray,
    agent: mbrl.planning.Agent,
    agent_kwargs: Dict,
    replay_buffer: ReplayBuffer,
    callback: Optional[Callable] = None,
    agent_uses_low_dim_obs: bool = False,
    optimizer_callback: Optional[Callable] = None,
) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """
    Overwrite of function step_env_and_add_to_buffer in mbrl.util.common
    """

    if agent_uses_low_dim_obs and not hasattr(env, "get_last_low_dim_obs"):
        raise RuntimeError(
            "Option agent_uses_low_dim_obs is only compatible with "
            "env of type mbrl.env.MujocoGymPixelWrapper."
        )
    if agent_uses_low_dim_obs:
        agent_obs = getattr(env, "get_last_low_dim_obs")()
    else:
        agent_obs = obs
    action = agent.act(agent_obs, optimizer_callback, **agent_kwargs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
    if callback:
        callback((obs, action, next_obs, reward, terminated, truncated))
    return next_obs, reward, terminated, truncated, info


def create_one_dim_tr_model_overriden(
    cfg: omegaconf.DictConfig,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """
    Overwrite of function create_one_dim_tr_model in mbrl.util.common
    """
    # This first part takes care of the case where model is BasicEnsemble and in/out sizes
    # are handled by member_cfg
    model_cfg = cfg.dynamics_model.model  # Changed from the original function
    if model_cfg._target_ == "mbrl.models.BasicEnsemble":
        model_cfg = model_cfg.member_cfg
    if model_cfg.get("in_size", None) is None:
        model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    if model_cfg.get("out_size", None) is None:
        model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)

    # Now instantiate the model
    model = hydra.utils.instantiate(model_cfg)  # Changed from the original function

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    else:
        obs_process_fn = None
    dynamics_model = hydra.utils.instantiate(
        cfg.dynamics_model.wrapper,
        model,
        target_is_delta=cfg.algorithm.target_is_delta,
        normalize=cfg.algorithm.normalize,
        normalize_double_precision=cfg.algorithm.get(
            "normalize_double_precision", False
        ),
        learned_rewards=cfg.algorithm.learned_rewards,
        obs_process_fn=obs_process_fn,
        no_delta_list=cfg.overrides.get("no_delta_list", None),
        num_elites=cfg.overrides.get("num_elites", None),
    )
    if model_dir:
        dynamics_model.load(model_dir)

    return dynamics_model


def get_run_kwargs(configs: omegaconf.DictConfig,) -> Dict:
    """
    Gather the important informations to initialize the wandb api

    :param configs: general configs (see configs directory)
    :return: arguments to iniate wandb
    """

    experiment_config = configs.experiment
    wandb_config = experiment_config.run_configs
    # Rename the run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    if wandb_config.group is not None:
        wandb_config.group = f"{wandb_config.group}_{sha}"
    else:
        wandb_config.name = f"{wandb_config.name}_{sha}"

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # TODO: Add pipeline recover from a checkpoint ? (Might be done in another process function)

    init_run_kwargs = omegaconf.OmegaConf.to_container(wandb_config, resolve=True)
    init_run_kwargs["config"] = omegaconf.OmegaConf.to_container(configs, resolve=True)

    return init_run_kwargs


def convert_yaml_config(
    config_path="configs/experiment/wandb.yaml",
    overrides: Dict = {},
    dictconfig: bool = True,
) -> Union[Dict, omegaconf.DictConfig]:
    """
    Convert a yaml file to a Dict or a DictConfig
    :param dictconfig: whether to return a Dict or a DictConfig, defaults to True
    """

    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Update with overrides
    config.update(overrides)

    # Create a DictConfig with omegaconf
    if dictconfig:
        return omegaconf.OmegaConf.create(config)

    return config
