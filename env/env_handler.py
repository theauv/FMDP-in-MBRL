from typing import Union, Optional, Dict, Tuple
import omegaconf
import hydra
import gymnasium as gym

import mbrl
from mbrl.util.env import EnvHandler, Freeze, _handle_learned_rewards_and_seed

from env.maze import ContinuousMaze
import env.termination_fns as term_fns
import env.reward_fns as rew_fns


class EnvironmentHandler(EnvHandler):

    freeze = Freeze

    @staticmethod
    def make_env(
        cfg: Union[Dict, omegaconf.ListConfig, omegaconf.DictConfig],
    ) -> Tuple[gym.Env, mbrl.types.TermFnType, Optional[mbrl.types.RewardFnType]]:

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
        else:
            return EnvHandler.make_env(cfg)

        env = gym.wrappers.TimeLimit(
            env, max_episode_steps=cfg.overrides.get("trial_length", 1000)
        )
        env, reward_fn = _handle_learned_rewards_and_seed(cfg, env, reward_fn)

        return env, term_fn, reward_fn
