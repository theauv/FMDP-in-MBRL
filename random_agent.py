import matplotlib
from matplotlib import pyplot as plt

import gymnasium as gym
import pygame
from env.hypergrid import ContinuousHyperGrid
from env.maze import ContinuousMaze

env = ContinuousHyperGrid(render_mode="human")
observation, info = env.reset(seed=42)

all_n_steps = []
all_rewards = []

n_steps = 0
rewards = 0
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    n_steps += 1
    rewards += reward

    if terminated or truncated:
        print("Yes")
        all_n_steps.append(n_steps)
        all_rewards.append(rewards)
        observation, info = env.reset()
        n_steps = 0
        rewards = 0
env.close()
pygame.display.quit()
pygame.quit()

matplotlib.use("TkAgg")
fig, axs = plt.subplots(1, 2)
axs[0].plot(all_n_steps)
axs[1].plot(all_rewards)
axs[0].set_xlabel("Episode")
axs[1].set_xlabel("Episode")
axs[0].set_ylabel("Number of steps")
axs[1].set_ylabel("Total reward")
plt.show()
