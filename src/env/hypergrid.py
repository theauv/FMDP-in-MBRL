from typing import Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab

from src.env.constants import *

matplotlib.use("Agg")


STEP_PENALTY = -1
GRID_DIM = 2
GRID_SIZE = 3.0  # Box(-GRID_SIZE, GRID_SIZE)
SIZE_END_BOX = 1.0  # Box in which the agent have to go to finish the game
STEP_SIZE = 1


class ContinuousHyperGrid(gym.Env):
    """Continuous high dimensional grid environment."""

    # WHAT IS METADATA ???
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        # Should be in a env_config (problem with term_fn and reward_fn)
        self.step_penalty = STEP_PENALTY
        self.grid_dim = GRID_DIM
        self.grid_size = GRID_SIZE
        self.size_end_box = (
            SIZE_END_BOX
        )  # Box in which the agent have to go to finish the game

        self.action_space = spaces.Box(
            -self.grid_size, self.grid_size, (self.grid_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-self.grid_size,
            high=self.grid_size,
            shape=(self.grid_dim,),
            dtype=np.float32,
        )

        self.all_distances = []
        self.state = None
        self.initial_box_size = 0.5

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

    def initial_state(self) -> np.array:
        """
        The initial states are all equally likely.
        They are distributed over the subspace [-self.initial_box_size, self.initial_box_size]^self.dim_grid
        :return: a random initial state
        """
        return np.random.rand(self.grid_dim)*self.initial_box_size - self.initial_box_size 

    def winning_state(self) -> bool:
        """
        :return: Whether the agent is in a winning position,
        the winning positions are a small subspace of the environment.
        Its size is defined by the attribute self.size_end_box
        """
        win = 1
        for i in range(self.grid_dim):
            win *= self.state[i] > (self.grid_size - self.size_end_box)

        return bool(win)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = action.squeeze()

        # Update state
        old_state = self.state
        self.state = (old_state + action + self.grid_size) % (
            self.grid_size * 2
        ) - self.grid_size  # High dim box is a closed world
        self.all_distances.append(
            sum(
                list(
                    map(
                        lambda x: max(0, self.grid_size - self.size_end_box - x),
                        self.state,
                    )
                )
            )
        )

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
        self.state = self.initial_state()  # Remark: extend to randomly starting point?
        self.steps_beyond_terminated = None
        self.all_distances.append(
            sum(
                list(
                    map(
                        lambda x: max(0, self.grid_size - self.size_end_box - x),
                        self.state,
                    )
                )
            )
        )
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

        if self.grid_dim == 2:
            # Continuous 2d space
            self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
            self.surf.fill(BLACK)

            #Plot starting square
            x = -self.initial_box_size * self.scale + self.offset
            y = self.initial_box_size * self.scale + self.offset
            starting_box = (
                (x, x),
                (y, x),
                (
                    y,
                    y,
                ),
                (x, y),
            )
            gfxdraw.polygon(self.surf, starting_box, WHITE)


            # Plot ending box
            x = self.grid_size * self.scale + self.offset
            y = (self.grid_size-self.size_end_box) * self.scale + self.offset
            end_box_wall = (
                (x, x),
                (y, x),
                (
                    y,
                    y,
                ),
                (x, y),
            )
            if self.winning_state():
                gfxdraw.filled_polygon(self.surf, end_box_wall, GREEN)
            else:
                gfxdraw.filled_polygon(self.surf, end_box_wall, RED)

            x, y = self.state * self.scale + self.offset
            gfxdraw.filled_circle(self.surf, int(x), int(y), 1, WHITE)

            self.surf = pygame.transform.flip(self.surf, False, True)
        else:
            # Plot plt graph of the l1 distance from the ending box
            fig = pylab.figure(
                figsize=[4, 4],  # Inches
                dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
            )
            ax = fig.gca()
            ax.plot(self.all_distances)
            ax.set_xlabel("step")
            ax.set_ylabel("Distance from finishing box (l1 norm)")

            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()

            size = canvas.get_width_height()

            self.surf = pygame.image.fromstring(raw_data, size, "RGB")

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
