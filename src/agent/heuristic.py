from typing import Optional, Union, List
from numbers import Number
import gymnasium as gym
import numpy as np
from scipy.spatial import distance
import geopy.distance

from mbrl.planning.core import Agent

from src.env.bikes import Bikes


class StubbornAgent(Agent):
    def __init__(
        self, env: gym.Env, action: Optional[Union[Number, List[Number]]] = None
    ) -> None:
        """
        Agent repeating the exact same action each environment step.
        :param action: The action can be directly pass to the agent
        (needs to be of the right dtype and dimension).
        Or you can pass a single value, then the whole action will be
        equal to this value for each action entry
        Or by default, None will sample a random action and fix it for the whole
        simulation.
        """
        if action is None:
            self.action = env.action_space.sample()
        elif hasattr(action, "__len__"):
            self.action = action
        else:
            # TODO: Maybe add "garde-fou", but should already exist in gym.Env
            self.action = np.ones(env.action_space.shape) * action

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """
        Return the same initialized action regardless of the observation
        """
        return self.action


class GoodBikesHeuristic(Agent):
    """
    TODO: Does not take into account taken_bikes
    TODO: Greedily solve each timeshift but not the whole day
    TODO: Does not take into account demand of a centroid being > bikes_per_truck
    TODO: Put as much as bikes as possible each time (does not try to minimize the number
    of bikes)
    """

    def __init__(self, env_config, env: Bikes) -> None:
        self.all_trips_data = env.all_trips_data
        self.action_timeshifts = env.action_timeshifts
        self.centroids = env.centroid_coords
        self.start_walk_dist_max = env_config.start_walk_dist_max
        self.end_walk_dist_max = env_config.end_walk_dist_max
        self.map_obs = env.map_obs
        self.map_act = env.map_act
        self.num_trucks = env_config.num_trucks
        self.bikes_per_truck = env_config.bikes_per_truck
        self.action_shape = env.action_space.shape

    def get_timeshift(self, counter):
        """
        get the current timeshift, namely the time between
        2 actions (the one we take now and the next one).

        :param counter: number of actions took so far. As the total
        number of actions per day is fixed we can find the timeshift from it.
        :return: current timeshift
        """
        if counter < len(self.action_timeshifts):
            return self.action_timeshifts[counter - 1 : counter + 1]
        elif counter >= len(self.action_timeshifts):
            return None

    def get_current_trips_data(self, month, day, time_counter):
        """
        Get the current trips data to forsee what will be the bikes dynamics
        at this current environment step
        """
        mask = (self.all_trips_data["Day"] == day) & (
            self.all_trips_data["Month"] == month
        )
        current_trips = self.all_trips_data[mask]

        # Keep only the trip at times we care:
        timeshift = self.get_timeshift(time_counter)
        start_time = timeshift[0]
        end_time = timeshift[1]
        time_mask = [
            float(time.replace(":", ".").split(".")[0]) >= start_time
            and float(time.replace(":", ".").split(".")[0]) <= end_time
            for time in current_trips["StartTime"].values
        ]
        current_trips = current_trips[time_mask]

        return current_trips

    def get_centroids_demand(self, bikes_distribution, current_trips):
        """
        Compute the bikes needed for each centroid to complete as many trips
        as possible

        :param bikes_distribution: _description_
        :param current_trips: _description_
        """

        demand = -np.array(bikes_distribution).copy()
        for i in range(len(current_trips)):
            trip = current_trips.iloc[i]
            start_loc = np.array([trip.StartLatitude, trip.StartLongitude])
            end_loc = np.array([trip.EndLatitude, trip.EndLongitude])
            starting_distances = distance.cdist(
                start_loc.reshape(-1, 2), self.centroids, metric="euclidean"
            )
            starting_centroid_idx_argsort = np.argsort(starting_distances)[0]
            ending_distances = distance.cdist(
                end_loc.reshape(-1, 2), self.centroids, metric="euclidean"
            )
            idx_end_centroid = np.argmin(ending_distances)

            for j, idx_start_centroid in enumerate(starting_centroid_idx_argsort):
                total_walking_distance = geopy.distance.distance(
                    start_loc, self.centroids[idx_start_centroid, :]
                ).km
                if total_walking_distance > self.start_walk_dist_max:
                    break
                else:
                    geo_ending_dist = geopy.distance.distance(
                        end_loc, self.centroids[idx_end_centroid, :]
                    ).km
                    if geo_ending_dist <= self.end_walk_dist_max:
                        demand[idx_start_centroid] += 1
        return demand

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        month = int(obs[self.map_obs["month"]][0])
        day = int(obs[self.map_obs["day"]][0])
        time_counter = int(obs[self.map_obs["time_counter"]][0])
        bikes_distribution = obs[self.map_obs["bikes_distr"]]

        current_trips = self.get_current_trips_data(
            month=month, day=day, time_counter=time_counter
        )
        demand = self.get_centroids_demand(
            bikes_distribution=bikes_distribution, current_trips=current_trips
        )
        best_centroids_idx = np.argpartition(demand, -self.num_trucks)[
            -self.num_trucks :
        ]

        action = np.zeros(self.action_shape)
        action[self.map_act["truck_centroid"]] = best_centroids_idx
        if "truck_num_bikes" in self.map_act.keys():
            action[self.map_act["truck_num_bikes"]] = self.bikes_per_truck

        return action


class ArtificialGoodBikesHeuristic(Agent):
    """
    TODO: Does not take into account taken_bikes
    TODO: Greedily solve each timeshift but not the whole day
    TODO: Does not take into account demand of a centroid being > bikes_per_truck
    TODO: Put as much as bikes as possible each time (does not try to minimize the number
    of bikes)
    """

    def __init__(self, env) -> None:
        self.get_demand = env.sim.trip_bikes
        self.time_step = env.sim.timestep
        self.action_timeshifts = env.action_timeshifts
        self.num_trucks = env.num_trucks
        self.bikes_per_truck = env.bikes_per_truck
        self.map_obs = env.map_obs
        self.map_act = env.map_act
        self.action_shape = env.action_space.shape
        self.num_centroids = env.num_centroids

    def get_timeshift(self, counter):
        """
        get the current timeshift, namely the time between
        2 actions (the one we take now and the next one).

        :param counter: number of actions took so far. As the total
        number of actions per day is fixed we can find the timeshift from it.
        :return: current timeshift
        """
        if counter < len(self.action_timeshifts):
            return self.action_timeshifts[counter - 1 : counter + 1]
        elif counter >= len(self.action_timeshifts):
            return None

    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        timeshift = self.get_timeshift(int(obs[self.map_obs["time_counter"]][0]))
        time_chunks = np.arange(timeshift[0], timeshift[1], self.time_step)
        demands = np.zeros(self.num_centroids)
        for time in time_chunks:
            demand = self.get_demand(time)[0]
            demands[demand] += 1

        best_centroids_idx = np.argpartition(demands, -self.num_trucks)[
            -self.num_trucks :
        ]

        action = np.zeros(self.action_shape)
        action[self.map_act["truck_centroid"]] = best_centroids_idx
        if "truck_num_bikes" in self.map_act.keys():
            action[self.map_act["truck_num_bikes"]] = self.bikes_per_truck

        return action


class CEMAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
