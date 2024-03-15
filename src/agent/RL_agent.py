from src.env.bikes import Bikes
from omegaconf import DictConfig
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback

from src.agent.heuristic import GoodBikesHeuristic

env_config = {
    "num_trucks": 5,  # 10
    "action_per_day": 8,
    "next_day_method": "random",  # sequential
    "initial_distribution": "zeros",
    "bikes_per_truck": 5,
    "start_walk_dist_max": 0.2,
    "end_walk_dist_max": 1000.0,
    "trip_duration": 0.5,
    "past_trip_data": "src/env/bikes_data/all_trips_LouVelo_merged.csv",
    "weather_data": "src/env/bikes_data/weather_data.csv",
    "centroids_coord": "src/env/bikes_data/LouVelo_centroids_coords.npy",
    "station_dependencies": "src/env/bikes_data/factors_radius_1-2.npy",
}
timesteps = 100000
env_config = DictConfig(env_config)
render_mode = None
env = Bikes(env_config, render_mode)

model = SAC.load("RL_agent/model/best_model.zip", env=env, verbose=1)
model = SAC("MlpPolicy", env, verbose=1)
eval_callback = EvalCallback(
    env,
    best_model_save_path="RL_agent/model",
    log_path="RL_agent/log",
    eval_freq=timesteps,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)
model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)

model = SAC.load("RL_agent/model/best_model.zip", env=env, verbose=1)
heuristic = GoodBikesHeuristic(env_config, env)

# vec_env = model.get_env()
obs, _ = env.reset()
rewards = []
step = 0
for j in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # action = heuristic.act(obs)
    obs, reward, done, truncated, info = env.step(action)
    print(action, "reward", reward)
    rewards.append(reward)
    # VecEnv resets automatically
    if done:
        print(
            "Episode reward: ",
            np.sum(rewards),
            np.mean(rewards),
            "episode steps: ",
            j - step,
            len(rewards),
        )
        obs, _ = env.reset()
        rewards = []
        step = j

env.close()
