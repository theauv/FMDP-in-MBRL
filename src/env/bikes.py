"""
Environments for the bikes experiments.
"""

import math
from typing import Optional, Dict, Tuple, List
from random import uniform

import geopy.distance
import gymnasium as gym
from gymnasium import logger, spaces
import numpy as np
from numpy.core.multiarray import array as array
import omegaconf
import pandas as pd
import pygame
import pygame.gfxdraw as gfxdraw
import pylab
from scipy.spatial import distance
import torch
import warnings

from src.env.constants import *
from src.env.dict_spaces_env import DictSpacesEnv


class Bikes(DictSpacesEnv):
    # WHAT IS METADATA ???
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(
        self,
        env_config: Optional[omegaconf.DictConfig],
        render_mode: Optional[str] = None,
        print_info: bool = True,
    ) -> None:
        # TODO: self.split_reward_by_centroid = env_config.split_reward_by_centroid # if true, we model the number of trips from each centroid individual, instead of the total number of trips
        # TODO: not especially optimal, could also decide for fix number of actions WHEN to take them during the day ??
        # TODO: "self strips" ?? Should we consider them or not (only influence is the trip_duration)
        # TODO: Use a structured np.array instead of map_obs and map_act dictionaries
        super().__init__()
        self.num_trucks = env_config.num_trucks
        self.bikes_per_truck = env_config.bikes_per_truck
        self.n_bikes = self.num_trucks * self.bikes_per_truck
        self.state = None
        self.data_looped = 0
        self.steps_beyond_terminated = None
        self.action_per_day = env_config.action_per_day
        self.day_start = 0.0
        self.day_end = 24.0
        self.action_timeshifts = list(
            np.linspace(self.day_start, self.day_end, self.action_per_day + 1)
        )
        self.next_day_method = env_config.next_day_method
        self.initial_distribution = env_config.initial_distribution
        hour_max = 24

        # TODO: rewrite this in a modulable way
        self.base_dir = env_config.get("base_dir", "")

        # Centroids data
        if env_config.centroids_coord is None:
            centroid_latitudes = [38.21, 38.27]
            centroid_longitudes = [-85.79, -85.71]
            centroid_coords = np.random.random((1000, 2)) * np.array(
                [
                    centroid_latitudes[1] - centroid_latitudes[0],
                    centroid_longitudes[1] - centroid_longitudes[0],
                ]
            ) + np.array([centroid_latitudes[0], centroid_longitudes[0]])
        else:
            centroid_coords = np.load(self.base_dir + env_config.centroids_coord)

        self.centroid_coords = centroid_coords
        centroids_idx = env_config.get("centroids_idx", None)
        if centroids_idx is not None:
            if hasattr(centroids_idx, "__len__"):
                self.centroid_coords = self.centroid_coords[centroids_idx]
            else:
                self.centroid_coords = self.centroid_coords[:centroids_idx]

        self.num_centroids = len(self.centroid_coords)

        if self.num_centroids > 100 or centroids_idx is not None:
            self.latitudes = [38.15, 38.35]
            self.longitudes = [-85.9, -85.55]
        else:
            self.latitudes = [38.2, 38.28]
            self.longitudes = [-85.8, -85.7]

        # Trips (and weather) data
        if env_config.past_trip_data is not None:
            assert (
                env_config.weather_data is not None
            ), "If using past_trip_data you must use a weather data too"
            self.all_trips_data = pd.read_csv(
                self.base_dir + env_config.past_trip_data,
                usecols=lambda x: x
                not in ["TripID", "StartDate", "EndDate", "EndTime"],
            )
            self.all_weather_data = pd.read_csv(
                self.base_dir + env_config.weather_data,
                usecols=[
                    "Year",
                    "Month",
                    "Day",
                    "DayOfWeek",
                    "Temp_Avg",
                    "Precip",
                    "Holiday",
                ],
            )

            # TODO: Keep week-end ?
            period = (
                "Month > 0 & Month < 13 & Year == 2019 & DayOfWeek >1 and DayOfWeek <=8"
            )
            area = (
                f"StartLatitude < {self.latitudes[1]} & StartLatitude > {self.latitudes[0]} & StartLongitude < {self.longitudes[1]} & StartLongitude > {self.longitudes[0]} "
                f"& EndLatitude < {self.latitudes[1]} & EndLatitude > {self.latitudes[0]} & EndLongitude < {self.longitudes[1]} & EndLongitude > {self.longitudes[0]} "
            )
            query = (
                "TripDuration < 60 & TripDuration > 0 & HourNum <= "
                + str(hour_max)
                + ""
                "&" + area + " & " + period
            )
            self.all_trips_data = self.all_trips_data.query(query)
            self.all_weather_data = self.all_weather_data.query(
                period + "& Holiday == 0"
            )
            assert len(self.all_trips_data) > 0
            assert len(self.all_weather_data) > 0
            self.max_demands = self.get_max_demands()
        else:
            self.all_trips_data = None
            self.all_weather_data = None

        # Station dependencies data (induced sparsity)
        self.station_dependencies_file = env_config.get("station_dependencies", None)
        station_dependencies = (
            np.load(self.base_dir + self.station_dependencies_file)
            if self.station_dependencies_file is not None
            else None
        )
        if (
            self.station_dependencies_file is not None
            and len(station_dependencies) != self.num_centroids
        ):
            raise ValueError(
                "Given station dependencies file does not match the number of centroids"
            )

        # Rentals simulator
        if self.all_trips_data is None:
            self.sim = ArtificialRentals_Simulator(
                self.centroid_coords,
                start_walk_dist_max=env_config.start_walk_dist_max,
                end_walk_dist_max=env_config.end_walk_dist_max,
                station_dependencies=station_dependencies,
                trip_duration=env_config.trip_duration,
                n_hubs_per_rushhour=env_config.get("n_hubs_per_rushhour", 1),
                n_edge_per_hub=env_config.get("n_edge_per_hub", 5),
                rushhour_std=env_config.get("rushhour_std", 1),
            )
        else:
            self.sim = Rentals_Simulator(
                self.centroid_coords,
                start_walk_dist_max=env_config.start_walk_dist_max,
                end_walk_dist_max=env_config.end_walk_dist_max,
                station_dependencies=station_dependencies,
                trip_duration=env_config.trip_duration,
                # TODO: Not sure if good idea to give shift_duration but induce more sparsity
            )

        # State-Action space
        # TODO: temperature, precipitation and demand in obs space ??
        obs_space = {
            "bikes_distr": spaces.Box(
                low=0,
                high=self.n_bikes * self.action_per_day,
                shape=(self.num_centroids,),
                dtype=np.float32,
            ),
            "day": spaces.Box(low=1, high=31, shape=(1,), dtype=np.float32),
            "day_of_week": spaces.Box(low=1, high=7, shape=(1,), dtype=np.float32),
            "month": spaces.Box(low=1, high=12, shape=(1,), dtype=np.float32),
            "time_counter": spaces.Box(
                low=1, high=self.action_per_day + 1, shape=(1,), dtype=np.float32
            ),
            "tot_n_bikes": spaces.Box(
                low=0,
                high=self.n_bikes * self.action_per_day,
                shape=(1,),
                dtype=np.float32,
            ),
        }
        if self.all_trips_data is not None:
            obs_space["demands"] = spaces.Box(
                low=0, high=1, shape=(self.action_per_day,), dtype=np.float32
            )
        self.dict_observation_space = spaces.Dict(obs_space)
        # TODO: At least time_counter would really make sense to be discrete (one-hot)
        # Explore possibility to handle heterogeneous spaces as input state

        action_space = {
            "truck_centroid": spaces.Box(
                low=0,
                high=self.num_centroids - 1,
                shape=(self.num_trucks,),
                dtype=np.float32,
            ),
        }
        if not env_config.fix_bikes_per_truck:
            action_space["truck_num_bikes"] = spaces.Box(
                low=0,
                high=self.bikes_per_truck,
                shape=(self.num_trucks,),
                dtype=np.float32,
            )
        self.dict_action_space = spaces.Dict(action_space)
        # TODO: could have a negative number of bikes meaning that we remove some bikes
        # in reward the more we have unused bikes the worst it is

        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        screen_ydim = 650
        screen_xdim = int(
            screen_ydim
            * abs(
                (self.longitudes[1] - self.longitudes[0])
                / (self.latitudes[0] - self.latitudes[1])
            )
        )
        self.screen_dim = (screen_xdim, screen_ydim)
        self.scale = np.abs(
            self.screen_dim
            / np.array(
                [
                    self.longitudes[1] - self.longitudes[0],
                    self.latitudes[0] - self.latitudes[1],
                ]
            )
        )
        self.offset = np.array([self.longitudes[0], self.latitudes[0]])
        self.screen = None
        self.clock = None
        self.isopen = True

        # Env info
        if print_info:
            self.get_env_info()

    def rescale_obs(self, flat_obs, reverse=False, keys: Optional[List[str]] = None):
        if keys is not None:
            raise NotImplementedError

        tot_n_bikes = flat_obs[..., self.map_obs["tot_n_bikes"]]

        for key, value in self.map_obs.items():
            if key != "length":
                if key == "bikes_distr":
                    high = tot_n_bikes
                else:
                    high = self.dict_observation_space[key].high
                    high = torch.tensor(high) if torch.is_tensor(flat_obs) else high
                low = self.dict_observation_space[key].low
                low = torch.tensor(low) if torch.is_tensor(flat_obs) else low

                if reverse:
                    flat_obs[..., value] = (high - low) * flat_obs[..., value] + low
                else:
                    flat_obs[..., value] = (flat_obs[..., value] - low) / (high - low)
        return flat_obs

    def flatten_obs(self, obs=None):
        return super().flatten_obs(obs)

    def flatten_action(self, action=None):
        return super().flatten_action(action)

    def embed1d_act(self, flat_act):
        if torch.is_tensor(flat_act):
            flat_act = flat_act.detach().numpy()
        flat_act = np.round(flat_act).astype(int)
        embedding = [
            centroid * (self.num_centroids) ** i
            for i, centroid in enumerate(flat_act[self.map_act["truck_centroid"]])
        ]
        if "truck_num_bikes" in self.map_act:
            embedding += [
                n_bikes * (self.bikes_per_truck) ** (i + self.num_trucks)
                for i, n_bikes in enumerate(flat_act[self.map_act["truck_num_bikes"]])
            ]
        return sum(embedding)

    def get_timeshift(self, state=None):
        if state is None:
            state = self.state
        counter = int(state["time_counter"])
        if counter < len(self.action_timeshifts):
            return self.action_timeshifts[counter - 1 : counter + 1]
        elif counter >= len(self.action_timeshifts):
            return None

    def get_env_info(self):
        print("-------------ENV INFO-------------")
        print(f"Observation space keys: {self.dict_observation_space.keys()}")
        print(f"Action space keys: {self.dict_action_space.keys()}")
        if self.all_trips_data is not None:
            print(
                f"Total number of used days in the year 2019: {len(self.all_weather_data)}"
            )
            print(f"Total number of trips: {len(self.all_trips_data)}")
            print(f"Max demand for each time shift: {self.max_demands}")
        else:
            print("Using artificial rentals dynamics")
        print(f"Number of centroids: {self.num_centroids}")
        print(f"Centroid induced dependencies file: {self.station_dependencies_file}")

    def get_max_demands(self):
        assert self.all_trips_data is not None, "Makes sense only using trips data"

        max_demands = np.zeros((self.action_per_day,))

        for i in range(self.all_weather_data.shape[0]):
            day = self.all_weather_data.iloc[i].Day
            month = self.all_weather_data.iloc[i].Month
            mask = (self.all_trips_data["Day"] == day) & (
                self.all_trips_data["Month"] == month
            )
            current_trips = self.all_trips_data[mask]

            for j in range(0, self.action_per_day):
                timeshift = self.action_timeshifts[j : j + 2]
                start_time = timeshift[0]
                end_time = timeshift[1]
                time_mask = [
                    float(time.replace(":", ".").split(".")[0]) >= start_time
                    and float(time.replace(":", ".").split(".")[0]) <= end_time
                    for time in current_trips["StartTime"].values
                ]
                demand = len(current_trips[time_mask])
                max_demands[j] = max(max_demands[j], demand)

        assert not np.any(max_demands <= 0)
        return max_demands

    def get_demands(self, day, month):
        demands = np.zeros((self.action_per_day,))

        mask = (self.all_trips_data["Day"] == day) & (
            self.all_trips_data["Month"] == month
        )
        current_trips = self.all_trips_data[mask]

        for j in range(0, self.action_per_day):
            timeshift = self.action_timeshifts[j : j + 2]
            start_time = timeshift[0]
            end_time = timeshift[1]
            time_mask = [
                float(time.replace(":", ".").split(".")[0]) >= start_time
                and float(time.replace(":", ".").split(".")[0]) <= end_time
                for time in current_trips["StartTime"].values
            ]
            demands[j] = len(current_trips[time_mask])

        return demands / self.max_demands

    def trips_steps(self, x=None):
        if x is None:
            x = self.state

        timeshift = self.get_timeshift(x)
        start_time = timeshift[0]
        end_time = timeshift[1]

        if self.all_trips_data is not None:
            mask = (self.all_trips_data["Day"] == x["day"]) & (
                self.all_trips_data["Month"] == x["month"]
            )
            current_trips = self.all_trips_data[mask]
            time_mask = [
                float(time.replace(":", ".").split(".")[0]) >= start_time
                and float(time.replace(":", ".").split(".")[0]) <= end_time
                for time in current_trips["StartTime"].values
            ]
            current_trips = current_trips[time_mask]

            # Compute the new state and all relevant informations about trips occuring during this timeshift
            (
                new_bikes_dist_after_shift,
                self.tot_num_trips,
                self.feasible_trips,
                self.num_met_trips,
                self.tot_demand_per_centroid,
                self.met_trips_per_centroid,
                self.adjacency,
            ) = self.sim.simulate_rentals(current_trips, x["bikes_distr"])

        else:
            (
                new_bikes_dist_after_shift,
                self.tot_num_trips,
                self.feasible_trips,
                self.num_met_trips,
                self.tot_demand_per_centroid,
                self.met_trips_per_centroid,
                self.adjacency,
            ) = self.sim.simulate_rentals(x["bikes_distr"], start_time, end_time)

        # TODO: Here we have an incremental reward computed at each time-step, but might make sense
        # to have just a final one (more accurate by harder for the RL agent to interpret)
        reward = self.compute_reward()

        return (new_bikes_dist_after_shift, reward)

    def compute_reward(self) -> float:
        # TODO: Ideas
        # trips_met/(tot_trips-too_far_from_centroid)
        # mean(met_demand_percentroid/tot_demand_per_centroid)
        # add the number of added bikes as a penalty
        # add the gasoil comsuption for each bike refill
        # warning if too_far_from_centroid > 0
        # Remark, adding the possibility to get add or remove bikes is somewhat equivalent to rebalancing

        # if self.tot_num_trips - self.feasible_trips > 0:
        #     warnings.warn(
        #         f"We have {self.tot_num_trips - self.feasible_trips} trips that could not met because there was no centroid close enough"
        #     )

        # SIMPLIEST REWARD
        reward = self.num_met_trips / max(1, self.feasible_trips)

        # PENALIZE ALSO PER CENTROID RATIO
        # relevant_trips = max(1,self.feasible_trips)
        # centroid_ratio = np.where(
        #     self.tot_demand_per_centroid != 0,
        #     self.met_trips_per_centroid / self.tot_demand_per_centroid,
        #     np.nan,
        # )
        # centroid_ratio = 0
        # if not np.all(self.tot_demand_per_centroid==0.):
        #     centroid_ratio = np.divide(self.met_trips_per_centroid, self.tot_demand_per_centroid, out=np.zeros(self.num_centroids), where=self.tot_demand_per_centroid!=0)
        #     centroid_ratio = np.nanmean(centroid_ratio)
        # reward = self.num_met_trips / relevant_trips + centroid_ratio

        # # TODO:param in init
        # DO NOT DELETE THIS !! (Might be a better reward proxy)
        # alpha = 0.7
        # beta = 0.7
        # min_bikes = 0.0

        # delta = (
        #     self.tot_demand_per_centroid
        #     - self.state["bikes_dist_before_shift"]
        #     + min_bikes
        # )
        # pos_delta_idx = np.where(delta > 0, True, False)

        # reward_1 = 0
        # if np.any(pos_delta_idx):
        #     reward_1 = 2 * alpha * np.minimum(
        #         self.delta_bikes[pos_delta_idx], delta[pos_delta_idx]
        #     ) - 2 * (1 - alpha) * np.maximum(
        #         0, self.delta_bikes[pos_delta_idx] - delta[pos_delta_idx]
        #     )
        # reward_2 = 0
        # if not np.all(pos_delta_idx):
        #     reward_2 = -self.delta_bikes[~pos_delta_idx]
        # reward = beta * np.mean(reward_1) + (1 - beta) * np.mean(reward_2)
        # reward = beta * np.mean(reward_1) + (1 - beta) * np.mean(reward_2)

        return reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # TODO: If only add bikes, we can exceed the upper bound of the obs space
        # But if we deal with it, we also need to be careful when simulating the rents,
        # Indeed, we should also check if the receiving centroid is able to store one more bike
        # But this is complicated if we care about the time duration ???

        if type(action) == np.ndarray:
            act = np.round(action)
            act = spaces.unflatten(self.dict_action_space, act)

        # Compute the number of new bikes added to each centroid
        self.delta_bikes = np.zeros(self.num_centroids, dtype=int)
        truck_centroid = act["truck_centroid"]
        truck_num_bikes = act.get(
            "truck_num_bikes", np.ones(self.num_centroids) * self.bikes_per_truck
        )
        for truck in range(self.num_trucks):
            self.delta_bikes[int(truck_centroid[int(truck)])] += truck_num_bikes[
                int(truck)
            ]

        # Update obs
        self.state["bikes_distr"] += self.delta_bikes
        self.state["tot_n_bikes"] += sum(self.delta_bikes)
        self.previous_bikes_distr = self.state["bikes_distr"]

        # Let all the vehicules being used during the day
        new_bikes_dist_after_shift, reward = self.trips_steps()
        self.state["bikes_distr"] = new_bikes_dist_after_shift

        # Render the environment
        if self.render_mode == "human":
            self.render()

        # Step the time counter
        self.state["time_counter"] += 1

        # Check if terminated
        terminated = self.get_timeshift() is None

        # Sanity check if we did not carry on after a finishing step
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

        return (
            spaces.flatten(self.dict_observation_space, self.state),
            reward,
            terminated,
            False,
            {},
        )  # observation, reward, end, truncated, info

    def get_initial_bikes_distribution(self) -> np.array:
        if self.initial_distribution == "uniform":
            x = np.zeros(self.num_centroids, dtype=int)
            for i, bike in enumerate(range(self.n_bikes)):
                x[i % self.num_centroids] += 1
        elif self.initial_distribution == "zeros":
            x = np.zeros(self.num_centroids, dtype=int)
        else:
            raise ValueError(
                f"There is no such initial bike distribution called {self.initial_distribution}"
            )

        return x

    def set_next_day_method(self, method: str):
        available_methods = ["random", "sequential"]
        if method in available_methods:
            self.next_day_method = method
        else:
            raise ValueError(
                f"Method {method} not available (available method: {available_methods})"
            )

    def get_next_day(self):
        # TODO: only weekday ??
        if self.next_day_method == "random":
            random_trip = self.all_weather_data.sample()
            day = random_trip["Day"].iloc[0]
            month = random_trip["Month"].iloc[0]
            day_of_week = random_trip["DayOfWeek"].iloc[0]
        elif self.next_day_method == "sequential":
            mask = (self.all_weather_data["Day"] == self.state["day"]) & (
                self.all_weather_data["Month"] == self.state["month"]
            )
            next_index = np.where(mask)[0][-1]
            if next_index + 1 >= self.all_weather_data.shape[0]:
                self.data_looped += 1
                next_index = 0
            next_trip = self.all_weather_data.iloc[next_index + 1]
            day = next_trip["Day"]
            month = next_trip["Month"]
            day_of_week = next_trip["DayOfWeek"]
        else:
            raise ValueError(
                f"No sample method named {self.next_day_method} implemented"
            )
        if self.data_looped > 0:
            warnings.warn(
                f"You are restarting the dataset (year) for the {self.data_looped}th time"
            )

        return day, month, day_of_week

    def reset(
        self, seed: Optional[int] = None, fix_day: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.all_trips_data is not None:
            if self.state is None:
                first_day = self.all_weather_data.iloc[0]
                day = first_day["Day"]
                month = first_day["Month"]
                day_of_week = first_day["DayOfWeek"]
            elif fix_day:
                day = self.state["day"]
                month = self.state["month"]
                day_of_week = self.state["day_of_week"]
            else:
                day, month, day_of_week = self.get_next_day()

            init_bikes_distr = self.get_initial_bikes_distribution()
            self.state = {
                "bikes_distr": init_bikes_distr,
                "day": day,
                "day_of_week": day_of_week,
                "month": month,
                "time_counter": 1,
                "demands": self.get_demands(day, month),
                "tot_n_bikes": sum(init_bikes_distr),
            }

        else:
            if self.state is None:
                # arbitrary
                day = 1
                month = 1
                day_of_week = 3
            else:
                day = self.state["day"] + 1 if self.state["day"] < 31 else 1
                month = (
                    self.state["month"]
                    if self.state["day"] < 31
                    else self.state["month"] + 1
                )
                day_of_week = (
                    self.state["day_of_week"] + 1
                    if self.state["day_of_week"] < 7
                    else 1
                )

            init_bikes_distr = self.get_initial_bikes_distribution()
            self.state = {
                "bikes_distr": init_bikes_distr,
                "day": day,
                "day_of_week": day_of_week,
                "month": month,
                "time_counter": 1,
                "tot_n_bikes": sum(init_bikes_distr),
            }

        self.tot_num_trips = None
        self.num_met_trips = None
        self.feasible_trips = None
        self.tot_demand_per_centroid = None
        self.met_trips_per_centroid = None
        self.adjacency = None
        self.delta_bikes = None
        self.previous_bikes_distr = None
        self.sim.reset()

        self.steps_beyond_terminated = None
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()

        return spaces.flatten(self.dict_observation_space, self.state), {}

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

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.screen_dim)
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface(self.screen_dim)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(self.screen_dim)
        self.surf.fill(BLACK)

        city_map = pygame.image.load(
            self.base_dir + "src/env/bikes_data/louisville_map.png"
        )
        city_size = city_map.get_size()
        city_real_dim = [-85.9, -85.55, 38.15, 38.35]
        city_scale = np.abs(
            city_size
            / np.array(
                [
                    city_real_dim[1] - city_real_dim[0],
                    city_real_dim[3] - city_real_dim[2],
                ]
            )
        )
        city_offset = np.array([city_real_dim[0], city_real_dim[2]])
        x = (self.longitudes - city_offset[0]) * city_scale[0]
        # Warning: pygame has a reversed y axis
        y = city_size[1] - (self.latitudes - city_offset[1]) * city_scale[1]
        cropped_region = (x[0], y[1], x[1] - x[0], y[0] - y[1])
        city_map = city_map.subsurface(cropped_region)
        city_map = pygame.transform.scale(city_map, self.screen_dim)
        self.surf.blit(city_map, (0, 0))

        # Added bikes from depot:
        font_size = 10
        font = pygame.font.SysFont("Arial", font_size)
        depot_coord = (self.screen_dim[0] - 50, self.screen_dim[1] - 50)
        if self.delta_bikes is not None:
            for i, added_bikes in enumerate(self.delta_bikes):
                if added_bikes > 0:
                    coord = self.centroid_coords[i]
                    coord = (coord[1], coord[0])
                    new_coord = (coord - self.offset) * self.scale
                    new_coord[1] = self.screen_dim[1] - new_coord[1]
                    width = 1  # added_bikes
                    self._draw_arrow(
                        self.surf,
                        pygame.Vector2(depot_coord[0], depot_coord[1]),
                        pygame.Vector2(new_coord[0], new_coord[1]),
                        PRETTY_RED,
                        width,
                        2 + min(5 * width, 10 + width),
                    )
                    txtsurf = font.render(str(added_bikes), True, DARK_RED)
                    alpha = uniform(0.25, 0.5)
                    text_coord = depot_coord + alpha * (new_coord - depot_coord)
                    self.surf.blit(
                        txtsurf,
                        (
                            text_coord[0] - font_size / 3.5,
                            text_coord[1] - font_size / 1.5,
                        ),
                    )
        pygame.draw.circle(self.surf, PRETTY_RED, depot_coord, 10)
        txtsurf = font.render("DEPOT", True, BLACK)
        self.surf.blit(
            txtsurf,
            (depot_coord[0] - font_size / 1.5, depot_coord[1] - font_size / 1.5),
        )

        # Centroids
        scale_bikes_render = 1 * (
            self.num_centroids / max(1, self.state["tot_n_bikes"] / 5)
        )
        offset_bikes_render = 5
        new_centroid_coords = []
        for coord, bikes in zip(self.centroid_coords, self.state["bikes_distr"]):
            coord = (coord[1], coord[0])
            new_coord = (coord - self.offset) * self.scale
            new_coord[1] = self.screen_dim[1] - new_coord[1]
            radius = offset_bikes_render + scale_bikes_render * bikes
            color = PRETTY_GREEN
            if radius >= 30:
                radius -= 30
                color = DARK_GREEN.copy()
                color[1] -= 15 * min(6, radius // 30)
                radius = 30
            pygame.draw.circle(self.surf, color, new_coord, radius)
            new_centroid_coords.append(new_coord)

        # Edges
        if self.adjacency is not None:
            for i, centroid_i in enumerate(self.adjacency):
                for j, centroid_j in enumerate(centroid_i):
                    if centroid_j > 0:
                        width = centroid_j
                        start_pos = (
                            self.centroid_coords[i][1],
                            self.centroid_coords[i][0],
                        )
                        start_pos = (start_pos - self.offset) * self.scale
                        start_pos[1] = self.screen_dim[1] - start_pos[1]
                        end_pos = (
                            self.centroid_coords[j][1],
                            self.centroid_coords[j][0],
                        )
                        end_pos = (end_pos - self.offset) * self.scale
                        end_pos[1] = self.screen_dim[1] - end_pos[1]
                        self._draw_arrow(
                            self.surf,
                            pygame.Vector2(start_pos[0], start_pos[1]),
                            pygame.Vector2(end_pos[0], end_pos[1]),
                            PRETTY_BLUE,
                            width,
                            2 + min(5 * width, 10 + width),
                        )

        # Demand bikes
        font_size = 10
        font = pygame.font.SysFont("Arial", font_size)
        if self.tot_demand_per_centroid is not None:
            for coord, demand in zip(new_centroid_coords, self.tot_demand_per_centroid):
                txtsurf = font.render(str(demand), True, BLACK)
                self.surf.blit(
                    txtsurf, (coord[0] - font_size / 3.5, coord[1] - font_size / 1.5)
                )

        # Legend:
        font_size = 15
        font = pygame.font.SysFont("Arial", font_size)
        shift = self.get_timeshift()
        title_str = (
            (
                f"Shift {shift[0]}:{shift[1]} Day: {int(self.state['day'])} "
                f"({int(self.state['day_of_week'])}/7) Month: {int(self.state['month'])}"
            )
            if shift is not None
            else (
                f"Shift: {shift} Day: {int(self.state['day'])} "
                f"({int(self.state['day_of_week'])}/7) Month: {int(self.state['month'])}"
            )
        )
        title = font.render(title_str, True, BLACK)
        self.surf.blit(title, (self.screen_dim[0] // 2, 0))
        pygame.draw.circle(self.surf, PRETTY_GREEN, (30, 20), offset_bikes_render)
        font_size = 10
        font = pygame.font.SysFont("Arial", font_size)
        demand = font.render("5", True, BLACK)
        self.surf.blit(demand, (30 - font_size / 3.5, 20 - font_size / 1.5))
        txtsurf = font.render(f"0 bikes, 5 demands", True, BLACK)
        self.surf.blit(txtsurf, (50, 20 - font_size // 2))
        n_bikes = int(15 / scale_bikes_render)
        pygame.draw.circle(
            self.surf,
            PRETTY_GREEN,
            (30, 50),
            offset_bikes_render + scale_bikes_render * n_bikes,
        )
        demand = font.render("3", True, BLACK)
        self.surf.blit(demand, (30 - font_size / 3.5, 50 - font_size / 1.5))
        txtsurf = font.render(f"{n_bikes} bikes, 3 demands", True, BLACK)
        self.surf.blit(txtsurf, (50, 50 - font_size // 2))
        self._draw_arrow(
            self.surf,
            pygame.Vector2(25, 80),
            pygame.Vector2(35, 80),
            PRETTY_BLUE,
            1,
            2 + min(5 * 1, 10),
        )
        txtsurf = font.render("1 bike", True, BLACK)
        self.surf.blit(txtsurf, (50, 80 - font_size // 2))
        self._draw_arrow(
            self.surf,
            pygame.Vector2(25, 110),
            pygame.Vector2(35, 110),
            PRETTY_BLUE,
            5,
            2 + min(5 * 5, 5 + 10),
        )
        txtsurf = font.render("5 bikes", True, BLACK)
        self.surf.blit(txtsurf, (50, 110 - font_size // 2))

        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def _draw_arrow(
        self,
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 2,
        head_width: int = 10,
        head_height: int = 5,
    ):
        """Draw an arrow between start and end with the arrow head at the end.

        Args:
            surface (pygame.Surface): The surface to draw on
            start (pygame.Vector2): Start position
            end (pygame.Vector2): End position
            color (pygame.Color): Color of the arrow
            body_width (int, optional): Defaults to 2.
            head_width (int, optional): Defaults to 4.
            head_height (float, optional): Defaults to 2.
        """
        arrow = start - end
        angle = arrow.angle_to(pygame.Vector2(0, -1))
        body_length = arrow.length() - head_height

        if body_width > 5:
            body_width -= 5
            color = DARK_BLUE.copy()
            color[1] -= 20 * min(5, body_width // 5)
            color[2] -= 30 * min(5, body_width // 5)
            body_width = 5

        if arrow.length() == 0:
            start.y -= 10
            radius = 10
            body_width = min(body_width, radius)
            gfxdraw.aacircle(surface, int(start.x), int(start.y), radius, color)
            gfxdraw.aacircle(
                surface, int(start.x), int(start.y), radius - body_width, color
            )
            pygame.draw.circle(
                surface, color, (int(start.x), int(start.y)), radius, body_width
            )
            # Create the triangle head around the origin
            head_verts = [
                pygame.Vector2(0, int(head_height / 2)),  # Center
                pygame.Vector2(
                    int(head_width / 2), -int(head_height / 2)
                ),  # Bottomright
                pygame.Vector2(
                    -int(head_width / 2), -int(head_height / 2)
                ),  # Bottomleft
            ]
            # Rotate and translate the head into place
            start.y -= radius
            translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(
                -angle
            )
            for i in range(len(head_verts)):
                head_verts[i].rotate_ip(-angle)
                head_verts[i] += translation
                head_verts[i] += start
            pygame.draw.polygon(surface, color, head_verts)
        else:
            # Create the triangle head around the origin
            head_verts = [
                pygame.Vector2(0, head_height / 2),  # Center
                pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
                pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
            ]
            # Rotate and translate the head into place
            translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(
                -angle
            )
            for i in range(len(head_verts)):
                head_verts[i].rotate_ip(-angle)
                head_verts[i] += translation
                head_verts[i] += start
            pygame.draw.polygon(surface, color, head_verts)

            # Stop weird shapes when the arrow is shorter than arrow head
            if arrow.length() >= head_height:
                head_height = arrow.length() // 2
            # Calculate the body rect, rotate and translate into place
            body_verts = [
                pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
                pygame.Vector2(body_width / 2, body_length / 2),  # Topright
                pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
                pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
            ]
            translation = pygame.Vector2(0, int(body_length / 2)).rotate(-angle)
            for i in range(len(body_verts)):
                body_verts[i].rotate_ip(-angle)
                body_verts[i] += translation
                body_verts[i] += start

            pygame.draw.polygon(surface, color, body_verts)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def termination_fn(
        self, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        # Check that
        done = next_obs[:, self.map_obs["time_counter"]] > self.action_per_day
        return done

    # TODO: Unused, but is it actually useful ????
    def reward_fn(self, actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def obs_preprocess_fn(self, batch_obs: torch.Tensor, batch_action: torch.Tensor):
        """
        We only want the model to learn the bikes rentals dynamics.
        So we preprocess the observation to manually compute the new
        bike distribution after taken the given action

        :return: preprocessed observation
        """

        batch_action = np.round(batch_action)
        resize = False
        while batch_obs.ndim < 3:
            assert batch_action.ndim == batch_obs.ndim
            batch_obs = batch_obs[None, ...]
            batch_action = batch_action[None, ...]
            resize = True

        ensemble_size = batch_obs.shape[0]
        batch_size = batch_obs.shape[1]
        distr_size = len(batch_obs[0, 0, self.map_obs["bikes_distr"]])

        # Compute delta_bikes in a parallel way
        delta_bikes = np.zeros((ensemble_size, batch_size, distr_size), dtype=int)
        truck_centroids = batch_action[..., self.map_act["truck_centroid"]]
        if self.map_act.get("truck_num_bikes", None) is not None:
            truck_bikes = batch_action[..., self.map_act["truck_num_bikes"]]
        else:
            truck_bikes = np.ones_like(truck_centroids) * self.bikes_per_truck
        n = distr_size
        truck_centroids = np.reshape(
            truck_centroids, (truck_centroids.shape[0] * truck_centroids.shape[1], -1)
        )
        offset = np.arange(truck_centroids.shape[0])[..., None]
        truck_centroids_offset = truck_centroids + offset * n
        unq, inv = np.unique(truck_centroids_offset.ravel(), return_inverse=True)
        unq = unq.astype(int)
        sol = np.bincount(inv, truck_bikes.ravel())
        delta_bikes[
            unq // (batch_size * n),
            (unq % (batch_size * n)) // n,
            (unq % (batch_size * n)) % n,
        ] = sol

        if resize:
            delta_bikes = delta_bikes.reshape((batch_size, -1))
            batch_action = batch_action.reshape((batch_size, -1))
            batch_obs = batch_obs.reshape((batch_size, -1))

        # Update obs
        batch_obs[..., self.map_obs["bikes_distr"]] += delta_bikes
        batch_obs[..., self.map_obs["tot_n_bikes"]] += np.sum(delta_bikes, axis=-1)[
            ..., None
        ]

        return batch_obs

    def add_missing_bikes(self, bikes_distr, k):
        k = k.item()
        if k > 0:
            idx = torch.topk(
                bikes_distr - torch.round(bikes_distr), k, sorted=False
            ).indices
            bikes_distr[idx] += 1
        if k < 0:
            idx = torch.topk(
                torch.round(bikes_distr) - bikes_distr, abs(k), sorted=False
            ).indices
            bikes_distr[idx] -= 1
        return bikes_distr

    def repeat_along_dim(self, pred, delta_tot_bikes):
        if pred.ndim > 2:
            return torch.stack(
                [
                    self.repeat_along_dim(x_i, delta_tot_bikes[i])
                    for i, x_i in enumerate(torch.unbind(pred, dim=0), 0)
                ],
                dim=0,
            )
        else:
            return torch.stack(
                [
                    self.add_missing_bikes(x_i, delta_tot_bikes[i])
                    for i, x_i in enumerate(torch.unbind(pred, dim=0), 0)
                ],
                dim=0,
            )

    def obs_postprocess_fn(
        self,
        batch_new_obs: torch.Tensor,
    ):
        """
        As we only want to learn the rentals dynamics, the learnable model
        will only return the new bike distribution but we need to return the whole
        new state. So we manually process the rest of the obs -> new_obs.
        In our case, we only need to increment the time_counter
        :return: postprocessed new_observation
        """
        batch_new_obs[..., self.map_obs["time_counter"]] += 1

        tot_n_bikes = batch_new_obs[..., self.map_obs["tot_n_bikes"]]
        proba_distr = torch.nn.functional.softmax(
            batch_new_obs[..., self.map_obs["bikes_distr"]], dim=-1
        )

        new_distr = proba_distr * tot_n_bikes
        round_new_distr = torch.round(new_distr)
        pred_n_bikes = torch.sum(round_new_distr, axis=-1).unsqueeze(-1)
        delta_tot_bikes = (tot_n_bikes - pred_n_bikes).detach().numpy().astype(int)

        new_distr = self.repeat_along_dim(new_distr, delta_tot_bikes)
        new_distr = torch.round(new_distr)
        batch_new_obs[..., self.map_obs["bikes_distr"]] = new_distr

        return batch_new_obs

    def obs_postprocess_pred_proba(
        self,
        batch_new_obs: torch.Tensor,
    ):
        """
        As we only want to learn the rentals dynamics, the learnable model
        will only return the new bike distribution but we need to return the whole
        new state. So we manually process the rest of the obs -> new_obs.
        In our case, we only need to increment the time_counter
        :return: postprocessed new_observation
        """
        batch_new_obs[..., self.map_obs["time_counter"]] += 1

        tot_n_bikes = batch_new_obs[..., self.map_obs["tot_n_bikes"]]
        proba_distr = torch.nn.functional.softmax(
            batch_new_obs[..., self.map_obs["bikes_distr"]], dim=-1, dtype=torch.float64
        )
        new_distr = torch.zeros(proba_distr.shape)
        new_distr = self.proba_repeat_along_dim(new_distr, proba_distr, tot_n_bikes)
        batch_new_obs[..., self.map_obs["bikes_distr"]] = new_distr

        return batch_new_obs

    def compute_new_distr(self, bikes_distr, proba_distr, tot_n_bikes):
        proba_distr /= torch.sum(proba_distr)
        centroid_idx = np.random.choice(
            np.arange(self.num_centroids), int(tot_n_bikes), p=proba_distr
        )
        unq, counts = np.unique(centroid_idx, return_counts=True)
        bikes_distr[unq] += counts.astype(np.float32)
        return bikes_distr

    def proba_repeat_along_dim(self, bikes_distr, proba_distr, tot_n_bikes):
        if bikes_distr.ndim > 2:
            return torch.stack(
                [
                    self.proba_repeat_along_dim(x_i, proba_distr[i], tot_n_bikes[i])
                    for i, x_i in enumerate(torch.unbind(bikes_distr, dim=0), 0)
                ],
                dim=0,
            )
        else:
            return torch.stack(
                [
                    self.compute_new_distr(x_i, proba_distr[i], tot_n_bikes[i])
                    for i, x_i in enumerate(torch.unbind(bikes_distr, dim=0), 0)
                ],
                dim=0,
            )


class Rentals_Simulator:
    """Class used to simulate rentals in the city from historic data."""

    def __init__(
        self,
        centroids,
        start_walk_dist_max=1,
        end_walk_dist_max=1,
        station_dependencies: Optional[np.array] = None,
        trip_duration: Optional[float] = None,
    ):
        self.centroids = centroids
        self.num_centroids = len(centroids)
        self.start_walk_dist_max = start_walk_dist_max
        self.end_walk_dist_max = end_walk_dist_max
        self.taken_bikes = []
        self.station_dependencies = station_dependencies
        self.station_dependencies_ll = self.station_dependencies_linked_list()
        self.trip_duration = trip_duration if trip_duration else 0.5

    def reset(self):
        self.taken_bikes = []

    def station_dependencies_linked_list(self):
        if self.station_dependencies is None:
            return None
        else:
            return [
                [i for i, e in enumerate(station) if e > 0]
                for station in self.station_dependencies
            ]

    def simulate_rentals(self, trips, X):
        """
        Simulate daily rentals when X[i] bikes are positioned in each region i at
        the beginning of the day on daynum of month.
        Returns a list of the starting coordinates of the trips that were met and
        a list of the starting coordinates of the trips that were unmet.
        """
        new_x = np.array(X)
        # All trips
        tot_num_trips = len(trips)
        feasible_trips = np.zeros(len(trips), dtype=int)
        num_met_trips = 0

        # trips per centroid
        tot_demand_per_centroid = np.zeros(self.num_centroids, dtype=int)
        met_trips_per_centroid = np.zeros(self.num_centroids, dtype=int)

        adjacency_matrix = np.zeros((self.num_centroids, self.num_centroids), dtype=int)

        # TODO: maybe not a fixed time for every trip ?
        # TODO: Penser à si dans un meme timeshift velos finissent leur trip
        # Does it make sense for the current reward ? Does it make sense for what we are doing ?
        # BIKE_SPEED = 20  # km/h

        total_walking_distance = 0

        assert not self.taken_bikes

        for i in range(tot_num_trips):
            trip = trips.iloc[i]
            start_time = trip.StartTime

            # this is a str in 24 hour format. convert to a float in hours
            start_time = float(start_time[0:2]) + float(start_time[3:5]) / 60

            # check if any bikes that were in transit have completed there trip, and add them back to the system
            # Note that for now taken bikes is always empty for simplification all bikes finish there
            # trips at the end of the timeshift
            for bike in self.taken_bikes:
                if bike[0] <= start_time:
                    new_x[bike[1]] += 1
                    self.taken_bikes.remove(bike)

            # Find the potential starting centroids
            start_loc = np.array([trip.StartLatitude, trip.StartLongitude])
            end_loc = np.array([trip.EndLatitude, trip.EndLongitude])
            starting_distances = distance.cdist(
                start_loc.reshape(-1, 2), self.centroids, metric="euclidean"
            )
            starting_centroid_idx_argsort = np.argsort(starting_distances)[0]

            # We go through the centroids in order of closest to farthest and see if there is an
            # available bike at one of the centroids to make the trip
            for j, idx_start_centroid in enumerate(starting_centroid_idx_argsort):
                total_walking_distance = geopy.distance.distance(
                    start_loc, self.centroids[idx_start_centroid, :]
                ).km
                if total_walking_distance > self.start_walk_dist_max:
                    break
                # Find the potential ending
                if self.station_dependencies_ll is not None:
                    if not self.station_dependencies_ll[idx_start_centroid]:
                        break
                    potential_ending_centroids = self.centroids[
                        self.station_dependencies_ll[idx_start_centroid]
                    ]
                else:
                    potential_ending_centroids = self.centroids
                ending_distances = distance.cdist(
                    end_loc.reshape(-1, 2),
                    potential_ending_centroids,
                    metric="euclidean",
                )
                idx_end_centroid = np.argmin(ending_distances)
                if self.station_dependencies_ll:
                    idx_end_centroid = self.station_dependencies_ll[idx_start_centroid][
                        idx_end_centroid
                    ]
                geo_ending_dist = geopy.distance.distance(
                    end_loc, self.centroids[idx_end_centroid, :]
                ).km
                if geo_ending_dist <= self.end_walk_dist_max:
                    feasible_trips[i] = 1
                    tot_demand_per_centroid[idx_start_centroid] += 1
                    if new_x[idx_start_centroid] > 0:
                        # If the trip can be met, we update all of the relevent lists tracking trips and where bikes are
                        new_x[idx_start_centroid] -= 1
                        total_walking_distance += geo_ending_dist

                        # Compute the duration of the trip
                        # distance_trip = geopy.distance.distance(
                        #     self.centroids[idx_start_centroid, :],
                        #     self.centroids[idx_end_centroid, :],
                        # ).km
                        # if distance_trip == 0.0:
                        #     delta_t = uniform(0.2, 1)
                        # else:
                        #     delta_t = distance_trip / BIKE_SPEED + uniform(0.1, 0.2)

                        self.taken_bikes.append(
                            (start_time + self.trip_duration, idx_end_centroid)
                        )
                        adjacency_matrix[idx_start_centroid, idx_end_centroid] += 1

                        # All trips
                        num_met_trips += 1
                        # trips per centroid
                        met_trips_per_centroid[idx_start_centroid] += 1
                        break

        feasible_trips = sum(feasible_trips)
        for bike in self.taken_bikes:
            new_x[bike[1]] += 1
            self.taken_bikes.remove(bike)
        assert tot_num_trips >= feasible_trips
        assert num_met_trips <= feasible_trips
        assert np.all(tot_demand_per_centroid >= met_trips_per_centroid)
        return (
            new_x,
            tot_num_trips,
            feasible_trips,
            num_met_trips,
            tot_demand_per_centroid,
            met_trips_per_centroid,
            adjacency_matrix,
        )


class ArtificialRentals_Simulator(Rentals_Simulator):
    """Class used to simulate fake rentals in the city."""

    def __init__(
        self,
        centroids,
        start_walk_dist_max=1,
        end_walk_dist_max=1,
        station_dependencies: Optional[np.array] = None,
        trip_duration: Optional[float] = None,
        time_step: Optional[float] = 1,
        n_hubs_per_rushhour: Optional[int] = 1,
        n_edge_per_hub: Optional[int] = 5,
        rushhour_std: Optional[float] = 1,
        seed: Optional[int] = 0,
    ):
        super().__init__(
            centroids,
            start_walk_dist_max,
            end_walk_dist_max,
            station_dependencies,
            trip_duration,
        )
        np.random.seed(seed)
        self.std = 0.1  # Factor of random noise in the bike distribution
        self.threshold = 0.5
        self.timestep = time_step  # hours
        self.rush_hours = [4, 8, 12, 16, 20]  # [2, 4, 8, 10, 12, 16, 18, 22]
        self.n_hubs_per_rushhour = n_hubs_per_rushhour
        self.n_edge_per_hub = n_edge_per_hub
        self.rushhour_std = rushhour_std

        if self.station_dependencies is None:
            self.station_dependencies = np.ones(
                (self.num_centroids, self.num_centroids)
            )
        self.mu, self.intensity = self.compute_mu_ij()  # Fixed
        self.sigma = self.compute_sigma_ij()  # Fixed
        # self.intensity = self.compute_intensity() #Fixed

    @staticmethod
    def gaussian(t, mu, sig, intensity):
        """
        Gives the mean probability that a bike is send at time x,
        knowing the rush hour distribution of each centroid (defined by mu and sigma)

        :param t: current time
        :param mu: mean rush hour for each connection centroid_i -> centroid_j (2d array)
        :param sig: standard deviation of the rush hour (2d array)
        :param intensity: intensity of the connection centroid_i -> centroid_j
        """
        return intensity * np.exp(-np.power((t - mu) / sig, 2.0) / 2)

    def trip_bikes(self, time):
        mean = self.gaussian(time, self.mu, self.sigma, self.intensity)
        random = np.random.normal(mean, self.std)  # A checker
        demand = random > self.threshold
        demand = np.multiply(demand, self.station_dependencies)
        return np.nonzero(demand)

    def compute_mu_ij(
        self,
    ):
        """
        Gives the mean rush hour of each connection centroid_i -> crntroid_j
        mu_ij = normal(rush_hour, 1) and the rush_hour is randomly assigned for each
        receiving centroid.
        return: 2d array
        """
        dim = self.num_centroids
        a = np.empty((dim, dim))
        intensity = np.random.normal(0.3, 0.1, (dim, dim))
        indices = np.arange(dim)
        split_size = math.ceil(dim // len(self.rush_hours))
        np.random.shuffle(indices)
        for i in range(len(self.rush_hours)):
            idx = indices[int(split_size * i) : int(split_size * (i + 1))]
            a[idx] = self.rush_hours[
                i
            ]  # np.random.normal(rush_hours[i], 1, (len(idx), dim))
            intensity_idx = np.random.choice(idx, self.n_hubs_per_rushhour)
            for i in intensity_idx:
                hub_edges = (
                    self.station_dependencies_ll[i]
                    if self.station_dependencies_ll is not None
                    else np.arange(dim)
                )
                edge_idx = np.random.choice(hub_edges, self.n_edge_per_hub)
                intensity[i, edge_idx] = np.random.normal(
                    0.7, 0.1, (1, self.n_edge_per_hub)
                )
        # mu = np.round(a).T
        mu = np.round(a)
        return mu, intensity

    def compute_sigma_ij(self):
        """return sigma_ij n_centroids x n_centroids with each entry is how wide the rush hour is
        for this connection could depend on day of week but start with constant one
        """
        # sigma = np.random.uniform(0.1, 1, (self.num_centroids, self.num_centroids))
        sigma = np.ones((self.num_centroids, self.num_centroids)) * self.rushhour_std
        return sigma

    def compute_intensity(self):
        """Each centroid has an intensity value defining how much likely it is to
        send a bike to its connections.
        """
        dim = self.num_centroids
        n_big_hubs = int(0.1 * dim)
        n_medium_hubs = int(0.4 * dim)
        n_small_hubs = int(0.5 * dim)
        n_not_used = dim - n_big_hubs - n_medium_hubs - n_small_hubs
        intensity_big = np.random.normal(
            0.6, 0.1, dim
        )  # np.random.uniform(0.5, 1., size=n_big_hubs)
        intensity_medium = np.random.normal(
            0.5, 0.1, dim
        )  # np.random.uniform(0.5, 0.8, size=n_medium_hubs)
        intensity_small = np.random.normal(
            0.3, 0.1, dim
        )  # np.random.uniform(0.1, 0.4, size=n_small_hubs)
        intensity_not_used = np.zeros(n_not_used)
        intensity = np.concatenate(
            [intensity_big, intensity_medium, intensity_small, intensity_not_used]
        )
        # intensity = np.random.normal(0.5, 0.1, dim)
        if not np.all(self.station_dependencies == 1):
            best_hubs = np.sum(self.station_dependencies, axis=0)
            best_hubs_idx = np.argsort(-best_hubs)
            intensity = intensity[best_hubs_idx]
        else:
            np.random.shuffle(intensity)
        intensity = np.random.normal(intensity, 0.1, (dim, dim))  # .T
        # intensity = np.repeat(intensity[...,None], dim, axis=-1)
        # intensity = np.repeat(intensity[None, ...], dim, axis=0)
        return intensity

    def simulate_rentals(self, bikes_distr, start, end):
        """
        Simulate daily rentals when X[i] bikes are positioned in each region i at
        the beginning of the day on daynum of month.
        Returns a list of the starting coordinates of the trips that were met and
        a list of the starting coordinates of the trips that were unmet.
        """

        new_bikes_ditr = np.array(bikes_distr).copy()

        # All trips
        tot_num_trips = 0
        num_met_trips = 0

        # trips per centroid
        tot_demand_per_centroid = np.zeros(self.num_centroids, dtype=int)
        met_trips_per_centroid = np.zeros(self.num_centroids, dtype=int)

        adjacency_matrix = np.zeros((self.num_centroids, self.num_centroids), dtype=int)

        assert not self.taken_bikes

        for time in np.arange(start, end, self.timestep):
            for bike in self.taken_bikes:
                if bike[0] <= time:
                    new_bikes_ditr[bike[1]] += 1
                    self.taken_bikes.remove(bike)

            demands = self.trip_bikes(time)
            tot_num_trips += len(demands[0])
            for idx_start_centroid, idx_end_centroid in zip(demands[0], demands[1]):
                # for debug only
                if new_bikes_ditr[idx_start_centroid] > 0:
                    new_bikes_ditr[idx_start_centroid] -= 1

                    # Compute the duration of the trip
                    # distance_trip = geopy.distance.distance(
                    #     self.centroids[idx_start_centroid, :],
                    #     self.centroids[idx_end_centroid, :],
                    # ).km
                    # if distance_trip == 0.0:
                    #     delta_t = uniform(0.2, 1)
                    # else:
                    #     delta_t = distance_trip / BIKE_SPEED + uniform(0.1, 0.2)

                    self.taken_bikes.append(
                        (time + self.trip_duration, idx_end_centroid)
                    )
                    adjacency_matrix[idx_start_centroid, idx_end_centroid] += 1

                    # All trips
                    num_met_trips += 1
                    # trips per centroid
                    met_trips_per_centroid[idx_start_centroid] += 1
                else:
                    tot_demand_per_centroid[idx_start_centroid] += 1

        for bike in self.taken_bikes:
            new_bikes_ditr[bike[1]] += 1
        self.taken_bikes = []

        return (
            new_bikes_ditr,
            tot_num_trips,
            tot_num_trips,  # feasible trips but theu are all feasible
            num_met_trips,
            tot_demand_per_centroid,
            met_trips_per_centroid,
            adjacency_matrix,
        )
