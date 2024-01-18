from typing import Optional, Dict, Tuple

from gymnasium import logger, spaces
import numpy as np
import omegaconf

from src.env.hypergrid import ContinuousHyperGrid


class DBNHyperGrid(ContinuousHyperGrid):
    """Continuous high dimensional grid environment."""

    def __init__(
        self,
        env_config: Optional[omegaconf.DictConfig],
        render_mode: Optional[str] = None,
    ) -> None:
        self.step_penalty = env_config.step_penalty
        self.action_dim = env_config.action_dim
        self.action_lim = env_config.action_lim
        self.state_dim = env_config.state_dim
        self.state_lim = env_config.state_lim
        self.state_dbn = env_config.state_dbn
        self.action_dbn = env_config.action_dbn
        self.size_end_box = env_config.size_end_box
        self.function_name = env_config.function

        if isinstance(self.action_lim, (int, float)) and isinstance(
            self.state_lim, (int, float)
        ):
            self.grid_size = self.state_lim * 2
            self.grid_dim = self.state_dim
            self.action_space = spaces.Box(
                -self.action_lim, self.action_lim, (self.action_dim,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-self.state_lim,
                high=self.state_lim,
                shape=(self.state_dim,),
                dtype=np.float32,
            )
        else:
            raise ValueError(
                "Given state and/or action dimensions and/or limits not supported (yet)"
            )

        self.all_distances = []
        self.state = None

        self.render_mode = render_mode
        self.viewer = None  # Useless?
        self.screen_dim = 500
        self.bound = 13
        self.scale = self.screen_dim / (self.bound * 2)
        self.offset = self.screen_dim // 2
        self.screen = None
        self.clock = None
        self.isopen = True

        self.steps_beyond_terminated = None

        assert np.max(np.array(self.state_dbn)) <= self.state_dim
        assert np.max(np.array(self.action_dbn)) <= self.action_dim
        assert len(self.state_dbn) == len(self.action_dbn) == self.state_dim

    def function(self, state, action):
        new_states = np.empty_like(state)
        for new_state_idx, (state_factors, action_factors) in enumerate(
            zip(self.state_dbn, self.action_dbn)
        ):
            if self.function_name == "sum":
                new_states[new_state_idx] = sum(state[state_factors]) + sum(
                    action[action_factors]
                )
            else:
                raise ValueError(f"No function {self.function_name} implemented")

        new_states = (new_states + self.grid_size) % (
            self.grid_size
        ) - self.grid_size / 2

        return new_states

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = action.squeeze()

        # Update state
        old_state = self.state
        self.state = self.function(old_state, action)  # High dim box is a closed world
        self.all_distances.append(
            sum(
                list(
                    map(
                        lambda x: max(0, self.grid_size / 2 - self.size_end_box - x),
                        self.state,
                    )
                )
            )
        )

        # Check if this is a final state
        terminated = self.is_winning_state()

        # Reward: penalty for taking another step
        reward = self.step_penalty

        # Check if we did not carry on after a finishing step
        if terminated:
            if self.steps_beyond_terminated is None:
                self.steps_beyond_terminated = 0
            else:
                if self.steps_beyond_terminated >= 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned terminated = True. You "
                        "should always call 'reset()' once you receive 'terminated = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1

        # Render the environment
        if self.render_mode == "human":
            self.render()

        return (
            self.state,
            reward,
            terminated,
            False,
            {},
        )  # observation, reward, end, truncated, info
