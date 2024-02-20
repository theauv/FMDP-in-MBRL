"""
Abstract class to handle complex RL environments whose observation and action spaces
are gym.Dict spaces and need to be flattened to be handle in a model such as NN. 
"""

from abc import ABC, abstractmethod
from typing import Optional, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

# TODO: Have a look at structured np.array might simplify the mapping part


class DictSpacesEnv(ABC, gym.Env):
    def __init__(self,):
        # TODO: Use a structured np.array instead of map_obs and map_act dictionaries
        self._dict_observation_space = spaces.Dict({"abstract": spaces.MultiBinary(1)})
        self._dict_action_space = spaces.Dict({"abstract": spaces.MultiBinary(1)})

        self.observation_space.sample = self.sample_obs
        self.action_space.sample = self.sample_action

    @property
    def dict_observation_space(self):
        """
        The dict_observation_space is an instance of the class gymnasium.spaces.Dict.
        This is an ordered dictionary (alphabetic order)
        """
        return self._dict_observation_space

    @dict_observation_space.setter
    def dict_observation_space(self, value):
        if isinstance(value, spaces.Dict):
            self._dict_observation_space = value
        else:
            raise ValueError(
                f"Observation space must be {type(self.dict_observation_space)} not {type(value)}"
            )

    @property
    def dict_action_space(self):
        """
        The dict_action_space is an instance of the class gymnasium.spaces.Dict.
        This is an ordered dictionary (alphabetic order)
        """
        return self._dict_action_space

    @dict_action_space.setter
    def dict_action_space(self, value):
        if isinstance(value, spaces.Dict):
            self._dict_action_space = value
        else:
            raise ValueError(
                f"Action space must be {type(self.action_space)} not {type(value)}"
            )

    @property
    def observation_space(self):
        return self.flatten_obs()

    @property
    def action_space(self):
        return self.flatten_action()

    @property
    def map_obs(self):
        return self.get_mapping_dict(self.dict_observation_space)

    @property
    def map_act(self):
        return self.get_mapping_dict(self.dict_action_space)

    @staticmethod
    def get_mapping_dict(dict_space):
        mapping = {}
        length = 0
        for key, value in dict_space.items():
            new_length = value.shape[0]
            mapping[key] = slice(length, length + new_length)
            length += new_length
        mapping["length"] = length
        return mapping

    def rescale_obs(self, flat_obs, scale=False, keys: Optional[List[str]] = None):
        for key, value in self.map_obs.items():
            if keys is not None:
                if key not in keys:
                    continue
            if key != "length":
                low = self.dict_observation_space[key].low
                high = self.dict_observation_space[key].high
                if torch.is_tensor(flat_obs):
                    low = torch.tensor(low)
                    high = torch.tensor(high)
                if scale:
                    flat_obs[..., value] = (high - low) * flat_obs[..., value] + low
                else:
                    flat_obs[..., value] = (flat_obs[..., value] - low) / (high - low)
        return flat_obs

    def rescale_act(self, flat_act, scale=False, keys: Optional[List[str]] = None):
        for key, value in self.map_act.items():
            if keys is not None:
                if key not in keys:
                    continue
            if key != "length":
                low = self.dict_action_space[key].low
                high = self.dict_action_space[key].high
                if torch.is_tensor(flat_act):
                    low = torch.tensor(low)
                    high = torch.tensor(high)
                if scale:
                    flat_act[..., value] = (high - low) * flat_act[..., value] + low
                else:
                    flat_act[..., value] = (flat_act[..., value] - low) / (high - low)
        return flat_act

    @abstractmethod
    def flatten_obs(self, obs=None):
        if obs is None:
            return spaces.flatten_space(self.dict_observation_space)
        else:
            return spaces.flatten(self.dict_observation_space, obs)

    @abstractmethod
    def flatten_action(self, action=None):
        if action is None:
            return spaces.flatten_space(self.dict_action_space)
        else:
            return spaces.flatten(self.dict_action_space, action)

    def sample_action(self):
        """
        The flattened action does keep the order of the dictionary space (alphabetic order)
        :return: flattened action
        """
        action = self.dict_action_space.sample()
        action = self.flatten_action(action)
        return np.round(action)

    def sample_obs(self):
        """
        The flattened obs does keep the order of the dictionary space (alphabetic order)
        :return: flattened obs
        """
        obs = self.dict_observation_space.sample()
        obs = self.flatten_obs(obs)
        return np.round(obs)
