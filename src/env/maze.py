from typing import Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled

from src.env.constants import *


# TODO MAKE SURE EVERYTHING WORKS WITH REWARD AND TERMINATION FUNCTIONS


def get_intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> bool:
    """
    Get the intersection of [A, B] and [C, D]. Return False if segment don't cross.
    Note: could use library "geometry"

    :param A: Point of the first segment
    :param B: Point of the first segment
    :param C: Point of the second segment
    :param D: Point of the second segment
    :return: whether there is an intersection or not
    """
    det = (B[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (B[1] - A[1])
    if det == 0:
        # Parallel
        return False
    else:
        t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
        t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det
        if t1 > 1 or t1 < 0 or t2 > 1 or t2 < 0:
            # not intersect
            return False
        else:
            xi = A[0] + t1 * (B[0] - A[0])
            yi = A[1] + t1 * (B[1] - A[1])
            return True


class ContinuousMaze(gym.Env):
    """Continuous maze environment."""

    # WHAT IS METADATA ???
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(self, render_mode: Optional[str] = None) -> None:

        # Should be in a env_config
        self.step_penalty = -0.1
        self.step_size = 10  # Could also be learned

        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-12, high=12, shape=(2,), dtype=np.float32
        )

        self.all_state = []
        self.state = None
        self.step_size = 10

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

        self.walls = np.array(
            [
                [[-12.0, -12.0], [-12.0, 12.0]],
                [[-10.0, 8.0], [-10.0, 10.0]],
                [[-10.0, 0.0], [-10.0, 6.0]],
                [[-10.0, -4.0], [-10.0, -2.0]],
                [[-10.0, -10.0], [-10.0, -6.0]],
                [[-8.0, 4.0], [-8.0, 8.0]],
                [[-8.0, -4.0], [-8.0, 0.0]],
                [[-8.0, -8.0], [-8.0, -6.0]],
                [[-6.0, 8.0], [-6.0, 10.0]],
                [[-6.0, 4.0], [-6.0, 6.0]],
                [[-6.0, 0.0], [-6.0, 2.0]],
                [[-6.0, -6.0], [-6.0, -4.0]],
                [[-4.0, 2.0], [-4.0, 8.0]],
                [[-4.0, -2.0], [-4.0, 0.0]],
                [[-4.0, -10.0], [-4.0, -6.0]],
                [[-2.0, 8.0], [-2.0, 12.0]],
                [[-2.0, 2.0], [-2.0, 6.0]],
                [[-2.0, -4.0], [-2.0, -2.0]],
                [[0.0, 6.0], [0.0, 12.0]],
                [[0.0, 2.0], [0.0, 4.0]],
                [[0.0, -8.0], [0.0, -6.0]],
                [[2.0, 8.0], [2.0, 10.0]],
                [[2.0, -8.0], [2.0, 6.0]],
                [[4.0, 10.0], [4.0, 12.0]],
                [[4.0, 4.0], [4.0, 6.0]],
                [[4.0, 0.0], [4.0, 2.0]],
                [[4.0, -6.0], [4.0, -2.0]],
                [[4.0, -10.0], [4.0, -8.0]],
                [[6.0, 10.0], [6.0, 12.0]],
                [[6.0, 6.0], [6.0, 8.0]],
                [[6.0, 0.0], [6.0, 2.0]],
                [[6.0, -8.0], [6.0, -6.0]],
                [[8.0, 10.0], [8.0, 12.0]],
                [[8.0, 4.0], [8.0, 6.0]],
                [[8.0, -4.0], [8.0, 2.0]],
                [[8.0, -10.0], [8.0, -8.0]],
                [[10.0, 10.0], [10.0, 12.0]],
                [[10.0, 4.0], [10.0, 8.0]],
                [[10.0, -2.0], [10.0, 0.0]],
                [[12.0, -12.0], [12.0, 12.0]],
                [[-12.0, 12.0], [12.0, 12.0]],
                [[-12.0, 10.0], [-10.0, 10.0]],
                [[-8.0, 10.0], [-6.0, 10.0]],
                [[-4.0, 10.0], [-2.0, 10.0]],
                [[2.0, 10.0], [4.0, 10.0]],
                [[-8.0, 8.0], [-2.0, 8.0]],
                [[2.0, 8.0], [8.0, 8.0]],
                [[-10.0, 6.0], [-8.0, 6.0]],
                [[-6.0, 6.0], [-2.0, 6.0]],
                [[6.0, 6.0], [8.0, 6.0]],
                [[0.0, 4.0], [6.0, 4.0]],
                [[-10.0, 2.0], [-6.0, 2.0]],
                [[-2.0, 2.0], [0.0, 2.0]],
                [[8.0, 2.0], [10.0, 2.0]],
                [[-4.0, 0.0], [-2.0, 0.0]],
                [[2.0, 0.0], [4.0, 0.0]],
                [[6.0, 0.0], [8.0, 0.0]],
                [[-6.0, -2.0], [2.0, -2.0]],
                [[4.0, -2.0], [10.0, -2.0]],
                [[-12.0, -4.0], [-8.0, -4.0]],
                [[-4.0, -4.0], [-2.0, -4.0]],
                [[0.0, -4.0], [6.0, -4.0]],
                [[8.0, -4.0], [10.0, -4.0]],
                [[-8.0, -6.0], [-6.0, -6.0]],
                [[-2.0, -6.0], [0.0, -6.0]],
                [[6.0, -6.0], [10.0, -6.0]],
                [[-12.0, -8.0], [-6.0, -8.0]],
                [[-2.0, -8.0], [2.0, -8.0]],
                [[4.0, -8.0], [6.0, -8.0]],
                [[8.0, -8.0], [10.0, -8.0]],
                [[-10.0, -10.0], [-8.0, -10.0]],
                [[-4.0, -10.0], [4.0, -10.0]],
                [[-12.0, -12.0], [12.0, -12.0]],
            ]
        )

    def winning_state(self) -> bool:
        """
        :return: Whether the agent escaped the maze or not
        """
        if self.state[0] > 10.0 or self.state[0] < -10.0:
            return True
        if self.state[1] > 10.0 or self.state[1] < -10.0:
            return True
        else:
            return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = action.squeeze()

        # Update state
        old_state = self.state
        self.state = old_state + action * self.step_size
        self.all_state.append(self.state.copy())

        # Check if we crossed a wall
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], old_state, self.state)
            if intersection:
                self.state = old_state

        # Check if this is a final state
        terminated = self.winning_state()

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

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.state = np.zeros(2)  # Remark: extend to randomly starting point?
        self.steps_beyond_terminated = None
        self.all_state.append(self.state.copy())
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self, mode: str = "human"):

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame  # type: ignore
            from pygame import gfxdraw  # type: ignore
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill(BLACK)
        for pos in self.all_state[1:10]:
            x, y = pos * self.scale + self.offset
            gfxdraw.filled_circle(self.surf, int(x), int(y), 1, RED)
        x, y = self.state * self.scale + self.offset
        gfxdraw.filled_circle(self.surf, int(x), int(y), 1, GREEN)

        for wall in self.walls:
            x1, y1 = wall[0] * self.scale + self.offset
            x2, y2 = wall[1] * self.scale + self.offset
            gfxdraw.line(self.surf, int(x1), int(y1), int(x2), int(y2), WHITE)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame  # type: ignore

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
