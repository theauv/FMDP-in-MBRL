# This file contains an overriden Class ReplayBuffer from mbrl library
# Allowing the ReplayBuffer to deal with loading larger dataset than its capacity

import numpy as np
from pathlib import Path
from typing import Union, Optional

from mbrl.util.replay_buffer import ReplayBuffer


class ReplayBufferOverriden(ReplayBuffer):
    def load(self, load_dir: Union[Path, str], num_to_store: Optional[int] = None):
        """Loads transition data from a given directory.
        Args:
            load_dir (str): the directory where the buffer is stored.
        """
        path = Path(load_dir) / "replay_buffer.npz"
        data = np.load(path)
        num_stored = (
            num_to_store
            if num_to_store is not None
            else min(len(data["obs"]), self.capacity)
        )
        self.obs[:num_stored] = data["obs"][:num_stored]
        self.next_obs[:num_stored] = data["next_obs"][:num_stored]
        self.action[:num_stored] = data["action"][:num_stored]
        self.reward[:num_stored] = data["reward"][:num_stored]
        self.terminated[:num_stored] = data["terminated"][:num_stored]
        self.truncated[:num_stored] = data["truncated"][:num_stored]
        self.num_stored = num_stored
        self.cur_idx = self.num_stored % self.capacity
        if "trajectory_indices" in data and len(data["trajectory_indices"]):
            self.trajectory_indices = data["trajectory_indices"]
