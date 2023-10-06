import matplotlib
from matplotlib import pyplot as plt
import torch
import numpy as np
import omegaconf
import hydra


from src.env.env_handler import HandMadeEnvHandler


@hydra.main(config_path="../configs", config_name="main")
def run(cfg: omegaconf.DictConfig):

    # create env and random seed
    cfg.overrides.render_mode = "human"
    env, term_fn, reward_fn = HandMadeEnvHandler.make_env(cfg)
    print(env.render_mode)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

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
            all_n_steps.append(n_steps)
            all_rewards.append(rewards)
            observation, info = env.reset()
            n_steps = 0
            rewards = 0
    env.close()

    matplotlib.use("tkAgg")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(all_n_steps)
    axs[1].plot(all_rewards)
    axs[0].set_xlabel("Episode")
    axs[1].set_xlabel("Episode")
    axs[0].set_ylabel("Number of steps")
    axs[1].set_ylabel("Total reward")
    plt.show()

if __name__ == "__main__":

    run()
