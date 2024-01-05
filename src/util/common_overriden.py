import pathlib
from typing import Dict, Union, Tuple, Optional, Callable, List

import gymnasium as gym
import hydra
import omegaconf
import numpy as np


import mbrl
from mbrl.util.replay_buffer import ReplayBuffer
from mbrl.planning import Agent

from src.model.lasso_net import LassoModelTrainer
from src.env.bikes import Bikes
from src.util.util import get_base_dir_path


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


def step_env_and_add_to_buffer_overriden(
    env: gym.Env,
    obs: np.ndarray,
    agent: Agent,
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


def get_env_factors(cfg: omegaconf.DictConfig,):
    """Utilitary function to initiliaze the correct factors in case
    you use a general factored model
    :return: correct factors
    """
    model_cfg = cfg.dynamics_model.model
    if cfg.overrides.env == "maze":
        factors = [[0, 2], [1, 3]]
    if cfg.overrides.env == "hypergrid":
        grid_dim = cfg.overrides.env_config.grid_dim
        factors = [[output, output + grid_dim] for output in range(grid_dim)]
    elif cfg.overrides.env == "dbn_hypergrid":
        state_factors = cfg.overrides.env_config.state_dbn
        action_factors = cfg.overrides.env_config.action_dbn
        state_dim = len(state_factors)
        factors = state_factors
        action_factors = [
            [factor + state_dim for factor in out_factors]
            for out_factors in action_factors
        ]
        for output, factor in enumerate(factors):
            for action_factor in action_factors[output]:
                factor.append(action_factor)
    elif cfg.overrides.env == "bikes":
        station_dependencies = cfg.overrides.env_config.get(
            "station_dependencies", None
        )
        if station_dependencies is None:
            raise ValueError(
                "You are using a factored model without specifying any factors in the environment"
            )

        base_dir = get_base_dir_path()
        adjacency = np.load(base_dir + station_dependencies)
        factors = [
            [i for i, e in enumerate(station) if e != 0] for station in adjacency
        ]
        additional_scopes = [
            i + adjacency.shape[0] for i in range(3)
        ]  # TODO: Needs to be changed if we change the observation
        factors = [factor + additional_scopes for factor in factors]
    else:
        raise ValueError(
            "No factors implementation for this env, either use a non-factored model \
                         either implement a way to get the factor of this env in this function"
        )

    return factors


def create_one_dim_tr_model_overriden(
    cfg: omegaconf.DictConfig,
    env: gym.Env,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """
    Overwrite of function create_one_dim_tr_model in mbrl.util.common
    TODO: Very specific conditions to instanciate the model dynamics in case of 
    the bikes environment. Try to modularize that.
    """
    # This first part takes care of the case where model is BasicEnsemble and in/out sizes
    # are handled by member_cfg

    # TODO: Modularize this, for the moment not sure how not to have a specific condition
    env_is_bikes = False
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    if isinstance(base_env, Bikes):
        env_is_bikes = True

    model_cfg = cfg.dynamics_model.model  # Changed from the original function
    if model_cfg._target_ == "mbrl.models.BasicEnsemble":
        model_cfg = model_cfg.member_cfg
    if model_cfg.get("in_size", None) is None:
        if env_is_bikes:
            model_cfg.in_size = sum(
                [
                    value.shape[0]
                    for key, value in base_env.dict_observation_space.items()
                    if key != "bikes_dist_after_shift"
                ]
            )
        else:
            model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    if model_cfg.get("out_size", None) is None:
        if env_is_bikes:
            model_cfg.out_size = base_env.dict_observation_space[
                "bikes_dist_after_shift"
            ].shape[0] + int(cfg.algorithm.learned_rewards)
        else:
            model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)
    if "factors" in model_cfg.keys() and model_cfg.get("factors", None) is None:
        model_cfg.factors = get_env_factors(cfg)

    # Now instantiate the model
    model = hydra.utils.instantiate(model_cfg)  # Changed from the original function

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    else:
        obs_process_fn = None
    if env_is_bikes:
        dynamics_model = hydra.utils.instantiate(
            cfg.overrides.model_wrapper,
            model,
            map_obs=base_env.map_obs,
            map_act=base_env.map_act,
            rescale_obs=base_env.rescale_obs,
            rescale_act=base_env.rescale_act,
            obs_preprocess_fn=base_env.obs_preprocess_fn,
            obs_postprocess_fn=base_env.obs_postprocess_fn,
            target_is_delta=cfg.algorithm.target_is_delta,
            normalize=cfg.algorithm.normalize,
            normalize_double_precision=cfg.algorithm.get(
                "normalize_double_precision", False
            ),
            learned_rewards=cfg.algorithm.learned_rewards,
            no_delta_list=cfg.overrides.get("no_delta_list", None),
            num_elites=cfg.overrides.get("num_elites", None),
        )
    else:
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
