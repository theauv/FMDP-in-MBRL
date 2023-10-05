from typing import Union, Optional, Dict, Tuple
import omegaconf
import gymnasium as gym

import mbrl
from mbrl.util.env import EnvHandler, Freeze, _handle_learned_rewards_and_seed

from src.env.maze import ContinuousMaze
from src.env.hypergrid import ContinuousHyperGrid
import src.env.termination_fns as term_fns
import src.env.reward_fns as rew_fns


class EnvironmentHandler(EnvHandler):

    freeze = Freeze

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
