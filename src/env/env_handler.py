from typing import Union, Optional, Dict, Tuple
import omegaconf
import gymnasium as gym
import numpy as np
from warnings import warn

import mbrl
from mbrl.util.env import EnvHandler, Freeze, _handle_learned_rewards_and_seed

from src.env.maze import ContinuousMaze
from src.env.hypergrid import ContinuousHyperGrid
import src.env.termination_fns as term_fns
import src.env.reward_fns as rew_fns

#TODO: Take a look back to all of this implementation (partially and quickly implemented, needs review)
#Good enough for now

ENVSHANDMADE = ["maze", "hypergrid"]

def get_handler(cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig]) -> EnvHandler:
    """
    :param cfg: general configs (see configs directory)
    :return: The right EnvHandler associated to the environment requested by the configs
    """
    cfg = omegaconf.OmegaConf.create(cfg)
    env_name = cfg.overrides.env
    if env_name in ENVSHANDMADE:
        return HandMadeEnvHandler()
    else:
        warn("You are trying to load an environment outside of this local project.\
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
            term_fn = term_fns.maze
            reward_fn = rew_fns.maze
        elif cfg.overrides.env == "hypergrid":
            env = ContinuousHyperGrid(render_mode)
            term_fn = term_fns.hypergrid
            reward_fn = rew_fns.hypergrid
        else:
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
            raise ValueError("Only handmade Environment are supported by this EnvHandler")
    
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
            raise ValueError("Only handmade Environment are supported by this EnvHandler")
