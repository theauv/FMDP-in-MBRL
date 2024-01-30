from typing import Union, Optional, Dict, Tuple
import omegaconf
from omegaconf import OmegaConf, open_dict
import gymnasium as gym
import numpy as np
from warnings import warn

import mbrl
from mbrl.util.env import EnvHandler, Freeze, _handle_learned_rewards_and_seed
from src.util.util import get_base_dir_path

from src.env.maze import ContinuousMaze
from src.env.hypergrid import ContinuousHyperGrid
from src.env.dbn_hypergrid import DBNHyperGrid
from src.env.bikes import Bikes

# TODO: Take a look back to all of this implementation (partially and quickly implemented, needs review)
# Good enough for now

ENVSHANDMADE = ["maze", "hypergrid"]


def get_handler(
    cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig]
) -> EnvHandler:
    """
    :param cfg: general configs (see configs directory)
    :return: The right EnvHandler associated to the environment requested by the configs
    """
    cfg = omegaconf.OmegaConf.create(cfg)
    env_name = cfg.overrides.env
    if env_name in ENVSHANDMADE:
        return HandMadeEnvHandler()
    else:
        warn(
            "You are trying to load an environment outside of this local project.\
             If this was not intentional, make sure you did add your environment \
             name in ENVSHANDMADE list"
        )
        return mbrl.util.create_handler(cfg)


class HandMadeEnvFreeze(Freeze):
    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if not HandMadeEnvHandler.is_correct_env_type(env):
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def __enter__(self):
        self.state = HandMadeEnvHandler.get_current_state(self._env)

    def __exit__(self, *_args):
        HandMadeEnvHandler.set_env_state(self.state, self._env)


def _is_handmade_gym_env(env: gym.wrappers.TimeLimit) -> bool:
    env = env.unwrapped
    return isinstance(env, ContinuousMaze) or isinstance(env, ContinuousHyperGrid)


class HandMadeEnvHandler(EnvHandler):
    """
    Environment handler dealing with the gym environment implemented in scr.env
    """

    freeze = HandMadeEnvFreeze

    @staticmethod
    def make_env(
        cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig],
    ) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:
        """
        Creates a gym environment specified in the onfigs (either from this repo
        or loading from mbrl library)

        :param cfg: general configs (see configs directory)
        :return: environment, and the termination and reward functions associated
        """

        cfg = omegaconf.OmegaConf.create(cfg)
        render_mode = cfg.overrides.get("render_mode", None)
        if cfg.overrides.env == "cartpole_continuous":
            env = mbrl.env.cartpole_continuous.CartPoleEnv(render_mode)
            term_fn = mbrl.env.termination_fns.cartpole
            reward_fn = mbrl.env.reward_fns.cartpole
        elif cfg.overrides.env == "maze":
            env = ContinuousMaze(render_mode)
            term_fn = env.termination_fn
            reward_fn = env.reward_fn
        elif cfg.overrides.env == "hypergrid":
            env = ContinuousHyperGrid(cfg.overrides.env_config, render_mode)
            term_fn = env.termination_fn
            reward_fn = env.reward_fn
        elif cfg.overrides.env == "dbn_hypergrid":
            env = DBNHyperGrid(cfg.overrides.env_config, render_mode)
            term_fn = env.termination_fn
            reward_fn = env.reward_fn
        elif cfg.overrides.env == "bikes":
            # TODO: Not best way to do it, but still more or less robust
            env_config = cfg.overrides.env_config
            OmegaConf.set_struct(env_config, True)
            with open_dict(env_config):
                env_config.base_dir = get_base_dir_path()
            cfg.overrides.num_steps = (
                cfg.overrides.num_episodes * env_config.action_per_day
            )
            cfg.overrides.dataset_size = cfg.overrides.num_steps
            env = Bikes(cfg.overrides.env_config, render_mode)
            term_fn = env.termination_fn
            reward_fn = env.reward_fn
        else:
            warn(
                "You are loading an environment outside of the scope of this local project \
                 (see MBRL library)"
            )
            return EnvHandler.make_env(cfg)

        env = gym.wrappers.TimeLimit(
            env, max_episode_steps=cfg.overrides.get("trial_length", 1000)
        )
        env, reward_fn = _handle_learned_rewards_and_seed(cfg, env, reward_fn)

        return env, term_fn, reward_fn

    @staticmethod
    def is_correct_env_type(env: gym.wrappers.TimeLimit) -> bool:
        return _is_handmade_gym_env(env)

    @staticmethod
    def make_env_from_str(env_name: str) -> gym.Env:
        raise NotImplementedError

    @staticmethod
    def get_current_state(env: gym.wrappers.TimeLimit) -> Tuple:
        warn(
            "This function has not been implemented, if this message is shown make \
             sure you did the changes necessary"
        )
        if _is_handmade_gym_env(env):
            env = env.unwrapped
            print("get", env.state)
            return env.state
        else:
            raise ValueError(
                "Only handmade Environment are supported by this EnvHandler"
            )

    @staticmethod
    def set_env_state(state: Tuple, env: gym.wrappers.TimeLimit) -> None:
        warn(
            "This function has not been implemented, if this message is shown make \
             sure you did the changes necessary"
        )
        if _is_handmade_gym_env(env):
            env = env.unwrapped
            env.state = state
        else:
            raise ValueError(
                "Only handmade Environment are supported by this EnvHandler"
            )
