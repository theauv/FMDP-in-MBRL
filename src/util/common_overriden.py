import pathlib
from typing import Dict, Union, Tuple, Optional, Callable, Sequence, Type

import gymnasium as gym
import hydra
import omegaconf
import numpy as np


import mbrl
from mbrl.util.replay_buffer import ReplayBuffer
from mbrl.planning import Agent

from src.util.model_trainer import LassoModelTrainer
from src.env.dict_spaces_env import DictSpacesEnv
from src.util.replay_buffer import ReplayBufferOverriden
from src.util.util import get_env_factors


def train_model_and_save_model_and_data_overriden(
    model: mbrl.models.Model,
    model_trainer: mbrl.models.ModelTrainer,
    cfg: omegaconf.DictConfig,
    replay_buffer: ReplayBuffer,
    work_dir: Optional[Union[str, pathlib.Path]] = None,
    callback: Optional[Callable] = None,
    callback_sparsity: Optional[Callable] = None,
    debug: bool = False,
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
            debug=debug,
        )
    else:
        model_trainer.train(
            dataset_train,
            dataset_val=dataset_val,
            num_epochs=cfg.get("num_epochs_train_model", None),
            patience=cfg.get("patience", 1),
            improvement_threshold=cfg.get("improvement_threshold", 0.01),
            callback=callback,
            debug=debug,
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


def create_one_dim_tr_model_overriden(
    cfg: omegaconf.DictConfig,
    env: gym.Env,
    obs_shape: Tuple[int, ...],
    act_shape: Tuple[int, ...],
    model_dir: Optional[Union[str, pathlib.Path]] = None,
):
    """
    Overwrite of function create_one_dim_tr_model in mbrl.util.common
    Handles DictSpacesEnv and factored environments
    """
    env_has_dict_spaces = False
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    if isinstance(base_env, DictSpacesEnv):
        env_has_dict_spaces = True

    model_cfg = cfg.dynamics_model.model
    if model_cfg._target_ == "mbrl.models.BasicEnsemble":
        model_cfg = model_cfg.member_cfg
    if model_cfg.get("in_size", None) is None:
        if env_has_dict_spaces:
            for input_obs_key in cfg.overrides.model_wrapper.model_input_obs_key:
                obs_keys = list(base_env.dict_observation_space.keys())
                if input_obs_key not in obs_keys:
                    raise ValueError(
                        f"Please give existing input obs keys in the configs:"
                        f" {obs_keys}. "
                        f"Given key '{input_obs_key}' does not exist in {type(base_env)}"
                    )
            for input_act_key in cfg.overrides.model_wrapper.model_input_act_key:
                act_keys = list(base_env.dict_action_space.keys())
                if input_act_key not in act_keys:
                    raise ValueError(
                        f"Please give existing input act keys in the configs:"
                        f" {act_keys}. "
                        f"Given key '{input_act_key}' does not exist in {type(base_env)}"
                    )
            obs_in_size = sum(
                [
                    base_env.dict_observation_space[key].shape[0]
                    for key in cfg.overrides.model_wrapper.model_input_obs_key
                ]
            )
            act_in_size = sum(
                [
                    base_env.dict_action_space[key].shape[0]
                    for key in cfg.overrides.model_wrapper.model_input_act_key
                ]
            )
            model_cfg.in_size = obs_in_size + act_in_size
        else:
            model_cfg.in_size = obs_shape[0] + (act_shape[0] if act_shape else 1)
    if model_cfg.get("out_size", None) is None:
        if env_has_dict_spaces:
            for output_key in cfg.overrides.model_wrapper.model_output_key:
                if output_key not in obs_keys:
                    raise ValueError(
                        f"Please give existing output keys in the configs:"
                        f" {obs_keys}. "
                        f"Given key '{output_key}' does not exist in {type(base_env)}"
                    )
            model_cfg.out_size = sum(
                [
                    base_env.dict_observation_space[key].shape[0]
                    for key in cfg.overrides.model_wrapper.model_output_key
                ]
            ) + int(cfg.algorithm.learned_rewards)
        else:
            model_cfg.out_size = obs_shape[0] + int(cfg.algorithm.learned_rewards)
    if "factors" in model_cfg.keys() and model_cfg.get("factors", None) is None:
        model_cfg.factors = get_env_factors(cfg, env=env)

    # Now instantiate the model
    model = hydra.utils.instantiate(model_cfg)

    name_obs_process_fn = cfg.overrides.get("obs_process_fn", None)
    if name_obs_process_fn:
        obs_process_fn = hydra.utils.get_method(cfg.overrides.obs_process_fn)
    else:
        obs_process_fn = None
    if env_has_dict_spaces:
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
            rescale_input=cfg.algorithm.rescale_input,
            normalize=cfg.algorithm.normalize,
            normalize_double_precision=cfg.algorithm.get(
                "normalize_double_precision", False
            ),
            learned_rewards=cfg.algorithm.learned_rewards,
            no_delta_list=cfg.overrides.get("no_delta_list", None),
            num_elites=cfg.dynamics_model.get("num_elites", None),
        )
    else:
        dynamics_model = hydra.utils.instantiate(
            cfg.overrides.model_wrapper,
            model,
            target_is_delta=cfg.algorithm.target_is_delta,
            normalize=cfg.algorithm.normalize,
            normalize_double_precision=cfg.algorithm.get(
                "normalize_double_precision", False
            ),
            learned_rewards=cfg.algorithm.learned_rewards,
            obs_process_fn=obs_process_fn,
            no_delta_list=cfg.overrides.get("no_delta_list", None),
            num_elites=cfg.dynamics_model.get("num_elites", None),
        )
    if model_dir:
        dynamics_model.load(model_dir)

    # Overrides
    cfg.overrides.model_batch_size = cfg.dynamics_model.get(
        "batch_size", cfg.overrides.model_batch_size
    )

    return dynamics_model


def create_overriden_replay_buffer(
    cfg: omegaconf.DictConfig,
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    obs_type: Type = np.float32,
    action_type: Type = np.float32,
    reward_type: Type = np.float32,
    load_dir: Optional[Union[str, pathlib.Path]] = None,
    collect_trajectories: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> ReplayBuffer:
    """Creates a replay buffer from a given configuration.

    The configuration should be structured as follows::

        -cfg
          -algorithm
            -dataset_size (int, optional): the maximum size of the train dataset/buffer
          -overrides
            -num_steps (int, optional): how many steps to take in the environment
            -trial_length (int, optional): the maximum length for trials. Only needed if
                ``collect_trajectories == True``.

    The size of the replay buffer can be determined by either providing
    ``cfg.algorithm.dataset_size``, or providing ``cfg.overrides.num_steps``.
    Specifying dataset set size directly takes precedence over number of steps.

    Args:
        cfg (omegaconf.DictConfig): the configuration to use.
        obs_shape (Sequence of ints): the shape of observation arrays.
        act_shape (Sequence of ints): the shape of action arrays.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        load_dir (optional str or pathlib.Path): if provided, the function will attempt to
            populate the buffers from "load_dir/replay_buffer.npz".
        collect_trajectories (bool, optional): if ``True`` sets the replay buffers to collect
            trajectory information. Defaults to ``False``.
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.

    Returns:
        (:class:`mbrl.replay_buffer.ReplayBuffer`): the replay buffer.
    """
    dataset_size = (
        cfg.algorithm.get("dataset_size", None) if "algorithm" in cfg else None
    )
    if not dataset_size:
        dataset_size = cfg.overrides.num_steps
    maybe_max_trajectory_len = None
    if collect_trajectories:
        if cfg.overrides.trial_length is None:
            raise ValueError(
                "cfg.overrides.trial_length must be set when "
                "collect_trajectories==True."
            )
        maybe_max_trajectory_len = cfg.overrides.trial_length

    replay_buffer = ReplayBufferOverriden(
        dataset_size,
        obs_shape,
        act_shape,
        obs_type=obs_type,
        action_type=action_type,
        reward_type=reward_type,
        rng=rng,
        max_trajectory_length=maybe_max_trajectory_len,
    )

    if load_dir:
        load_dir = pathlib.Path(load_dir)
        replay_buffer.load(str(load_dir))

    return replay_buffer
