from typing import Optional, Dict, Tuple
from shapely import Point, LineString

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import matplotlib
import matplotlib.backends.backend_agg as agg
import numpy as np
import omegaconf
import pylab
import torch

from src.env.constants import *


class ContinuousHyperGrid(gym.Env):
    """Continuous high dimensional grid environment."""

    # WHAT IS METADATA ???
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(
        self,
        env_config: Optional[omegaconf.DictConfig],
        render_mode: Optional[str] = None,
    ) -> None:
        self.step_penalty = env_config.step_penalty
        self.grid_dim = env_config.grid_dim
        self.grid_size = env_config.grid_size
        self.size_end_box = env_config.size_end_box
        self.step_size = env_config.step_size

        assert self.step_size < self.grid_size

        self.action_space = spaces.Box(
            -self.step_size, self.step_size, (self.grid_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-self.grid_size / 2,
            high=self.grid_size / 2,
            shape=(self.grid_dim,),
            dtype=np.float32,
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

        # Obstacles
        self.n_obstacles = env_config.n_obstacles
        self.size_obstacles = env_config.size_obstacles
        self.obstacles = self.get_obstacles()

    def get_initial_state(self) -> np.array:
        """
        The initial states are all equally likely.
        They are distributed over the whole hyperspace (except the winning subspace)
        :return: a random initial state
        """

        # TODO: Rethink this function, not very optimal

        x = np.ones(self.grid_dim) * self.grid_size / 2
        intersection = False
        while self.is_winning_state(x) or intersection:
            x = self.observation_space.sample()
            intersection = False
            for obstacle_pos in self.obstacles:
                intersection_ = (
                    Point(obstacle_pos)
                    .buffer(self.size_obstacles / 2, cap_style="square")
                    .intersects(Point(x))
                )
                if intersection_:
                    intersection = True
                    break
        return x

    def is_winning_state(self, x: Optional[np.array] = None) -> bool:
        """
        :return: Whether the agent is in a winning position,
        the winning positions are a small subspace of the environment.
        Its size is defined by the attribute self.size_end_box
        """

        if x is None:
            x = self.state

        win = 1
        for i in range(self.grid_dim):
            win *= x[i] > (self.grid_size / 2 - self.size_end_box)

        return bool(win)

    def get_obstacles(self):
        obstacles = []
        if self.n_obstacles is None:
            return obstacles
        if self.n_obstacles <= 0 or self.grid_dim > 3 or self.grid_dim <= 1:
            return obstacles
        ending_point = [(self.grid_size / 2) - 0.5] * self.grid_dim
        for i in range(self.n_obstacles):
            obstacle_pos = self.observation_space.sample()
            intersection = (
                Point(obstacle_pos)
                .buffer(self.size_obstacles / 2, cap_style="square")
                .intersects(
                    Point(ending_point).buffer(
                        self.size_end_box / 2, cap_style="square"
                    )
                )
            )
            iter = 0
            max_iter = 10  # HARD-CODED
            while intersection and iter < max_iter:
                obstacle_pos = self.observation_space.sample()
                intersection = (
                    Point(obstacle_pos)
                    .buffer(self.size_obstacles / 2, cap_style="square")
                    .intersects(
                        Point(ending_point).buffer(
                            self.size_end_box / 2, cap_style="square"
                        )
                    )
                )
                iter += 1
            if iter < max_iter:
                obstacles.append(obstacle_pos)

        return obstacles

    def is_colliding(self, old_state, new_state=None):
        """
        :return: Whether the agent is in colliding with an obstacle
        """

        if new_state is None:
            new_state = self.state

        move = LineString([old_state, new_state])

        for obstacle_pos in self.obstacles:
            point = Point(obstacle_pos)
            obstacle = point.buffer(self.size_obstacles / 2, cap_style="square")
            collision = move.intersects(obstacle)
            if collision:
                return True
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        action = action.squeeze()

        # Update state
        old_state = self.state
        self.state = (old_state + action + self.grid_size / 2) % (
            self.grid_size
        ) - self.grid_size / 2  # High dim box is a closed world
        if self.obstacles:
            if self.is_colliding(old_state=old_state):
                self.state = old_state  # If action leads to a collision, don't move
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

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.state = (
            self.get_initial_state()
        )  # Remark: extend to randomly starting point?
        self.initial_state = self.state
        self.initial_distance = sum(
            list(
                map(
                    lambda x: max(0, self.grid_size / 2 - self.size_end_box - x),
                    self.initial_state,
                )
            )
        )
        self.all_distances = [self.initial_distance]
        self.steps_beyond_terminated = None
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self, mode: str = None):
        if mode is None:
            mode = self.render_mode

        if mode is None:
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
            if mode == "human":
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

            # Draw starting square
            init_box_size = 0.2
            x_1 = (self.initial_state[0] - init_box_size / 2) * self.scale + self.offset
            x_2 = (self.initial_state[0] + init_box_size / 2) * self.scale + self.offset
            y_1 = (self.initial_state[1] - init_box_size / 2) * self.scale + self.offset
            y_2 = (self.initial_state[1] + init_box_size / 2) * self.scale + self.offset
            starting_box = ((x_1, y_2), (x_1, y_1), (x_2, y_1), (x_2, y_2))
            gfxdraw.polygon(self.surf, starting_box, WHITE)

            # Draw ending box
            x = self.grid_size / 2 * self.scale + self.offset
            y = (self.grid_size / 2 - self.size_end_box) * self.scale + self.offset
            end_box_wall = ((x, x), (y, x), (y, y), (x, y))
            if self.is_winning_state():
                gfxdraw.filled_polygon(self.surf, end_box_wall, GREEN)
            else:
                gfxdraw.filled_polygon(self.surf, end_box_wall, RED)

            # Draw the obstacles
            if self.obstacles:
                for obstacle_position in self.obstacles:
                    x_1 = (
                        obstacle_position[0] - self.size_obstacles / 2
                    ) * self.scale + self.offset
                    x_2 = (
                        obstacle_position[0] + self.size_obstacles / 2
                    ) * self.scale + self.offset
                    y_1 = (
                        obstacle_position[1] - self.size_obstacles / 2
                    ) * self.scale + self.offset
                    y_2 = (
                        obstacle_position[1] + self.size_obstacles / 2
                    ) * self.scale + self.offset
                    obstacle = ((x_1, y_1), (x_2, y_1), (x_2, y_2), (x_1, y_2))
                    gfxdraw.filled_polygon(self.surf, obstacle, WHITE)

            # Draw the current position
            x, y = self.state * self.scale + self.offset
            gfxdraw.filled_circle(self.surf, int(x), int(y), 1, WHITE)

            self.surf = pygame.transform.flip(self.surf, False, True)
        else:
            matplotlib.use("Agg")
            # Plot plt graph of the l1 distance from the ending box
            fig = pylab.figure(
                figsize=[4, 4],  # Inches
                dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
            )
            ax = fig.gca()
            ax.plot(self.all_distances, "o")
            ax.plot(self.initial_distance, "ro")
            if self.is_winning_state():
                ax.plot(len(self.all_distances), self.all_distances[-1], "go")
            ax.set_xlabel("step")
            ax.set_ylabel("Distance from finishing box (l1 norm)")

            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()

            size = canvas.get_width_height()

            self.surf = pygame.image.fromstring(raw_data, size, "RGB")

        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame  # type: ignore

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def termination_fn(
        self, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Termination function associated to the hypergrid env

        :param action: batch of actions
        :param next_obs: batch of next_obs
        :return: batch of bool tensors whether the associated (action, next_obs)
                is a final state
        """
        assert len(next_obs.shape) == 2

        done = torch.ones(next_obs.shape[0], dtype=bool)
        for i in range(self.grid_dim):
            done *= next_obs[:, i] > (self.grid_size / 2 - self.size_end_box)

        done = done[:, None]  # augment dimension
        return done

    def reward_fn(self, action: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Reward function associated to the hypergrid env

        :param action: batch of actions
        :param next_obs: batch of next_obs
        :return: batch of rewards associated to each (action, next_obs)
        """
        assert len(next_obs.shape) == len(action.shape) == 2

        return (
            (~self.termination_fn(action, next_obs) * self.step_penalty)
            .float()
            .view(-1, 1)
        )
