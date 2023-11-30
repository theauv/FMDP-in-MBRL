"""
Environments for the bikes experiments.
I would first recommend looking at functions.py since the class structure is similar 
and the environments are much simpler. 
"""

from typing import Optional, Dict, Tuple

import geopy
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
import numpy as np
import omegaconf
import pandas as pd
import pickle
import pylab
from scipy.spatial import distance
import torch
import warnings


class Rentals_Simulator:
    """ Class used to simulate rentals in the city from historic data."""

    def __init__(self, trips_data, centroids, depth=2, walk_dist_max=1):
        self.trips_data = trips_data  # history of trips data (used to simulate rentals)
        self.centroids = (
            centroids
        )  # centroids for regions in the city where bikes/scooters are positioned
        self.R = len(
            centroids
        )  # number of regions in the city TODO: don't like the multiple definitions of R
        # self.chunks = depth +1 #number of time chunks to break the day into. You add 1 to the depth
        self.walk_dist_max = walk_dist_max

    def simulate_rentals(self, trips, X):
        """ 
        Simulate daily rentals when X[i] bikes are positioned in each region i at the beginning of the day on daynum of month.
        Returns a list of the starting coordinates of the trips that were met and a list of the starting coordinates of the trips that were unmet.
        """
        # query = 'Day == ' + str(daynum) + '& Month == ' + str(month)
        # daily_trips = self.trips_data.query(query)
        new_x = np.array(X)
        R = len(self.centroids)  # TODO: don't like the multiple definitions of R
        # All trips
        tot_num_trips = len(trips)
        num_met_trips = 0
        too_far_from_centroid_trips = 0

        # All trips coords
        trips_starting_coords = []
        met_trips_starting_coords = []
        too_far_from_centroid_trips_coords = []

        # trips per centroid
        tot_demand_per_centroid = np.zeros(R)
        met_trips_per_centroid = np.zeros(R)

        adjacency_matrix = np.zeros((R, R))

        # TODO: maybe not a fixed time for every trip ?
        TRIP_DURATION = (
            0.5
        )  # in hours: the time a bike is removed from the system for while a trip is happening

        taken_bikes = []
        total_walking_distance = 0

        for i in range(tot_num_trips):
            start_loc = np.array(
                [trips.iloc[i].StartLatitude, trips.iloc[i].StartLongitude]
            )
            start_time = trips.iloc[i].StartTime

            trips_starting_coords.append(start_loc)
            met_trips_starting_coords.append(None)

            # this is a str in 24 hour format. convert to a float in hours
            start_time = float(start_time[0:2]) + float(start_time[3:5]) / 60

            # check if any bikes that were in transit have completed there trip, and add them back to the system
            for bike in taken_bikes:
                if bike[0] <= start_time:
                    new_x[bike[1]] += 1
                    taken_bikes.remove(bike)

            distances = distance.cdist(
                start_loc.reshape(-1, 2), self.centroids, metric="euclidean"
            )
            idx_argsort = np.argsort(distances)[0]
            # We go through the centroids in order of closest to farthest and see if there is an available bike at one of the centroids to make the trip
            for idx_start_centroid in idx_argsort:
                walk_dist_max = self.walk_dist_max
                if (
                    geopy.distance.distance(
                        start_loc, self.centroids[idx_start_centroid, :]
                    ).km
                    > walk_dist_max
                ):
                    too_far_from_centroid_trips += 1
                    too_far_from_centroid_trips_coords.append(start_loc)
                    break
                if new_x[idx_start_centroid] > 0:
                    # If the trip can be met, we update all of the relevent lists tracking trips and where bikes are
                    new_x[idx_start_centroid] -= 1
                    end_loc = np.array(
                        [trips.iloc[i].EndLatitude, trips.iloc[i].EndLongitude]
                    )
                    distances = distance.cdist(
                        end_loc.reshape(-1, 2), self.centroids, metric="euclidean"
                    )
                    idx_end_centroid = np.argmin(distances)
                    total_walking_distance += geopy.distance.distance(
                        start_loc, self.centroids[idx_start_centroid, :]
                    ).km
                    taken_bikes.append((start_time + TRIP_DURATION, idx_end_centroid))

                    adjacency_matrix[idx_start_centroid, idx_end_centroid] += 1

                    # All trips
                    num_met_trips += 1

                    # All trips coords
                    met_trips_starting_coords[-1] = self.centroids[idx_start_centroid]

                    # trips per centroid
                    met_trips_per_centroid[idx_start_centroid] += 1
                    tot_demand_per_centroid[idx_start_centroid] += 1
                    break
                else:
                    tot_demand_per_centroid[idx_start_centroid] += 1

        return (
            new_x,
            tot_num_trips,
            num_met_trips,
            too_far_from_centroid_trips,
            trips_starting_coords,
            met_trips_starting_coords,
            too_far_from_centroid_trips_coords,
            tot_demand_per_centroid,
            met_trips_per_centroid,
            adjacency_matrix,
        )


class Bikes(gym):
    # WHAT IS METADATA ???
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(
        self,
        env_config: Optional[omegaconf.DictConfig],
        render_mode: Optional[str] = None,
    ) -> None:

        # N is the number of trucks
        self.num_trucks = env_config.num_trucks
        # TODO: self.split_reward_by_centroid = env_config.split_reward_by_centroid # if true, we model the number of trips from each centroid individual, instead of the total number of trips
        # TODO: not especially optimal, could also decide for fix number of actions WHEN to take them during the day ??
        self.action_per_day = env_config.action_per_day
        self.day_start = 0.0
        self.day_end = 24.0
        self.action_timeshifts = list(
            np.linspace(self.day_start, self.day_end, self.action_per_day + 1)
        )
        self.sample_method = env_config.sample_method

        # load centroids_trips_matrix5_40 from pckl. This is the number of trips betwen each pair of regions (normalized)
        # TODO: rewrite this in a modulable way
        self.centroid_trips_matrix = pickle.load(
            open("scripts/bikes_data/centroid_trips_matrix5_40.pckl", "rb")
        )
        # get the coordinates of the depots
        _, _, _, _, _, _, centroid_coords = pickle.load(
            open("scripts/bikes_data/training_data_" + "5" + "_" + "40" + ".pckl", "rb")
        )
        R = len(centroid_coords)
        self.num_centroids = (
            R if env_config.centroids is None else env_config.centroids
        )  # the number of bike depots

        self.centroid_coords = centroid_coords
        self.bikes_per_truck = (
            env_config.bikes_per_truck
        )  # 8 #The number of bikes dropped-off by each truck
        self.n_bikes = self.N * self.bikes_per_truck

        # TODO: Or should we fix the sequence -> Multidiscrete multidim np.array ?
        # TODO: normalize and how do we do with discrete values ???
        self.observation_space = spaces.Dict(
            {
                "bikes_distribution": spaces.Sequence(
                    np.ones(self.num_centroids) * self.n_bikes
                ),
                "day": spaces.Discrete(31, start=1),
                "month": spaces.Discrete(12, start=1),
                "timeshift": spaces.Box(self.day_start, self.day_end, shape=2),
            }
        )
        self.observation_space = spaces.Sequence(
            np.ones(self.num_centroids) * self.n_bikes
        )
        # self.action_space = spaces.Sequence(spaces.MultiDiscrete([self.bikes_per_truck, self.num_centroids, self.num_centroids]))
        self.action_space = spaces.Dict(
            {
                "truck_num_bikes": spaces.MultiDiscrete(
                    np.ones(self.num_trucks) * self.bikes_per_truck
                ),
                "truck_centroid": spaces.MultiDiscrete(
                    np.ones(self.num_trucks) * self.num_centroids
                ),
            }
        )
        # TODO: could have a negative number of bikes meaning that we remove some bikes
        # in reward the more we have unused bikes the worst it is

        self.all_trips_data = pd.read_csv(
            "scripts/bikes_data/dockless-vehicles-3_full.csv",
            usecols=lambda x: x not in ["TripID", "StartDate", "EndDate", "EndTime"],
        )
        self.all_weather_data = pd.read_csv(
            "scripts/bikes_data/weather_data.csv",
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

        self.hour_max = 24

        self.period = (
            "Month > 0 & Month < 13 & Year == 19 & DayOfWeek >=0 and DayOfWeek <=8"
        )
        self.area = (
            "StartLatitude < 38.35 & StartLatitude > 38.15 & StartLongitude < -85.55 & StartLongitude > -85.9 "
            "& EndLatitude < 38.35 & EndLatitude > 38.15 & EndLongitude < -85.55 & EndLongitude > -85.9 "
        )
        query = (
            "TripDuration < 60 & TripDuration > 0 & HourNum <= "
            + str(self.hour_max)
            + ""
            "&" + self.area + " & " + self.period
        )
        self.all_trips_data = self.all_trips_data.query(query)
        self.all_weather_data = self.all_weather_data.query(
            self.period + "& Holiday == 0"
        )

        # TODO: weather and demand do somehting with it later on !!!!!
        # take out all weather data that corresponds to a weekend. 1.0 is a sunday and 7.0 is a saturday in the dataset
        # self.weekdays = [2,3,4,5,6]
        # self.weather_data = weather_data.query('DayOfWeek in @weekdays')

        # self.weather_data = weather_data
        # size of the inputs not controlled by the truck (weather, demand)
        # self.z_shape = 3
        # if self.chunk_demand:
        #     self.z_shape = self.z_shape + self.depth +1

        # self.z_max = None

        # self.z_max = self.get_z_max()

        self.sim = Rentals_Simulator(
            self.all_trips_data,
            centroid_coords,
            depth=self.depth,
            walk_dist_max=env_config.walk_distance_max,
        )

        self.state = None

        self.render_mode = render_mode
        self.viewer = None
        self.screen_dim = 500
        self.bound = 13
        self.scale = self.screen_dim / (self.bound * 2)
        self.offset = self.screen_dim // 2
        self.screen = None
        self.clock = None
        self.isopen = True

        self.steps_beyond_terminated = None

    def trips_steps(self):
        mask = (self.all_trips_data["Day"] == self.state["day"]) & (
            self.all_trips_data["Month"] == self.state["month"]
        )
        current_trips = self.all_trips_data[mask]

        # Keep only the trip at times we care:
        start_time = self.state["timeshift"][0]
        end_time = self.state["timeshift"][1]

        time_mask = [
            float(time.replace(":", ".").split(":")[0]) >= start_time
            and float(time.replace(":", ".").split(":")[0]) <= end_time
            for time in current_trips["StartTime"].values
        ]

        current_trips = current_trips[time_mask]

        # Compute the new state and all relevant informations about trips occuring during this timeshift
        (
            new_bikes_distribution,
            tot_num_trips,
            num_met_trips,
            too_far_from_centroid_trips,
            trips_starting_coords,
            met_trips_starting_coords,
            too_far_from_centroid_trips_coords,
            self.tot_demand_per_centroid,
            self.met_trips_per_centroid,
            self.adjacency,
        ) = self.sim.simulate_rentals(current_trips, self.state)

        reward = self.compute_reward(
            tot_num_trips,
            num_met_trips,
            too_far_from_centroid_trips,
            self.tot_demand_per_centroid,
            self.met_trips_per_centroid,
        )

        return new_bikes_distribution, reward

    def set_new_state(self, new_bikes_distribution):

        self.state["bikes_distribution"] = new_bikes_distribution

        new_index = self.action_timeshifts.index(self.state["timeshift"][1])
        if new_index < len(self.action_timeshifts) - 1:
            self.state["timeshift"] = self.action_timeshifts[new_index : new_index + 2]
        else:
            raise ValueError("The episode should be already finished !!")

    def compute_reward(
        self,
        tot_num_trips,
        num_met_trips,
        too_far_from_centroid_trips,
        tot_demand_per_centroid,
        met_trips_per_centroid,
    ) -> float:
        # TODO: Ideas
        # trips_met/(tot_trips-too_far_from_centroid)
        # mean(met_demand_percentroid/tot_demand_per_centroid)
        # add the number of added bikes as a penalty
        # add the gasoil comsuption for each bike refill
        # warning if too_far_from_centroid > 0

        # Remark, adding the possibility to get add or remove bikes is somewhat equivalent to rebalancing

        warnings.warn(
            f"We have {too_far_from_centroid_trips} trips that could not met because there was no centroid close enough"
        )

        return num_met_trips / (tot_num_trips - too_far_from_centroid_trips) + np.mean(
            met_trips_per_centroid / tot_demand_per_centroid
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:

        # TODO: model could be a GNN
        action = action.squeeze()
        self.action = action

        # Update state
        old_state = self.state

        delta_bikes = np.zeros(self.num_centroids)
        truck_centroid = action["truck_centroid"]
        truck_num_bikes = action["truck_num_bikes"]
        for truck in self.num_trucks:
            delta_bikes[truck_centroid[truck]] += truck_num_bikes[truck]

        self.state["bikes_distribution"] += delta_bikes

        # Render the environment
        if self.render_mode == "human":
            self.render()

        # Let all the vehicules being used during the day
        state_before = self.state
        new_bikes_distribution, reward = self.trips_steps()
        self.set_new_state(new_bikes_distribution)

        # Check if end of the day
        terminated = self.state["timeshift"][1] == self.day_end

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

    def get_initial_bikes_distribution(self, distribution="uniform") -> np.array:
        """
        The initial states are all equally likely.
        They are distributed over the whole hyperspace (except the winning subspace)
        :return: a random initial state
        """
        # TODO: initiate graph with same number of vehicules in each region OR random.
        if distribution == "uniform":
            bikes_per_region = self.n_bikes / self.num_centroids
            x = np.ones(self.num_centroids) * bikes_per_region
        else:
            raise ValueError(
                f"There is no such initial bike distribution called {distribution}"
            )

        return x

    def new_day(self):
        # TODO: only weekday ??
        if self.sample_method == "random":
            random_trip = self.all_trips_data.sample()
            day = random_trip["Day"]
            month = random_trip["Month"]
        elif self.sample_method == "sequential":
            next_trip = self.all_trips_data[
                self.all_trips_data.Day != self.state["day"]
            ].head(1)
            day = next_trip["Day"]
            month = next_trip["Month"]
        else:
            raise ValueError(f"No sample method named {self.sample_method} implemented")

        return day, month

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.state is None:
            first_day = self.all_trips_data.iloc[0]
            day = first_day["Day"]
            month = first_day["Month"]
        else:
            day, month = self.new_day()
        self.state = {
            "bikes_per_centroid": self.get_initial_bikes_distribution(),
            "day": day,
            "month": month,
            "timeshift": self.action_timeshifts[:2],
        }

        self.tot_demand_per_centroid = None
        self.met_trips_per_centroid = None
        self.adjacency = None
        self.action = None

        self.steps_beyond_terminated = None
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def render(self, mode: str = None):

        # TODO

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

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill(BLACK)

        # DO SOMETHING
        if self.action is None:
            # only centroid
            pass
        else:
            pass

        self.surf = pygame.transform.flip(self.surf, False, True)

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


class Bikes(FNEnv):
    """
    A basic bikes class that is used as a base for the BikesSparse class used in our experiments. It is significantly more general than BikesSparse. 
    Does not contain an evaluate function because it is only meant to be subclassed.
    """

    def __init__(
        self,
        depth=2,
        centroids=116,
        N=5,
        chunk_demand=True,
        alpha=0.01,
        split_reward_by_centroid=True,
        walk_distance_max=1.0,
    ):
        # N is the number of trucks
        self.N = N
        self.depth = (
            depth
        )  # depth is the depth of the graph. If depth is 1, each node is the total trips for a given region in the day. If depth is two, we have seperate nodes for before and after noon, etc.
        self.centroids = centroids  # the number of bike depots
        self.split_reward_by_centroid = (
            split_reward_by_centroid
        )  # if true, we model the number of trips from each centroid individual, instead of the total number of trips

        self.alpha = (
            alpha
        )  # threshold for determining graph edges from the trip data. If the number of trips from region i to region j is greater than alpha, then there is an edge between trips from region i at time t to j at time t+1

        # load centroids_trips_matrix5_40 from pckl. This is the number of trips betwen each pair of regions (normalized)
        self.centroid_trips_matrix = pickle.load(
            open("scripts/bikes_data/centroid_trips_matrix5_40.pckl", "rb")
        )

        self.parent_nodes = self.get_parent_nodes()
        self.bikes_per_truck = 8  # the number of bikes dropped-off by each truck
        self.n_bikes = self.N * self.bikes_per_truck
        self.chunk_demand = (
            chunk_demand
        )  # whether to use a finer granularity when reporting demand information (demand per centroid) or just the total demand

        dag = DAG(self.parent_nodes)

        # self.input_dim_a = 1
        active_input_indices, full_input_indices = self.adapt_active_input_indices()
        self.active_input_indices = active_input_indices

        self.input_dim_a = N

        action_input = FNActionInput(active_input_indices)
        full_input = FNActionInput(full_input_indices)

        discrete = (
            2
        )  # doesn't matter because we have discrete player input and continuous adversary input, so we end up overriding this

        noise_scales = 0.0

        super(Bikes, self).__init__(
            dag,
            action_input,
            discrete=discrete,
            input_dim_a=self.input_dim_a,
            full_input=full_input,
        )

        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

        trips_data = pd.read_csv(
            "scripts/bikes_data/dockless-vehicles-3_full.csv",
            usecols=lambda x: x not in ["TripID", "StartDate", "EndDate", "EndTime"],
        )
        weather_data = pd.read_csv(
            "scripts/bikes_data/weather_data.csv",
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

        hour_max = 24

        period = "Month > 0 & Month < 13 & Year == 19 & DayOfWeek >=0 and DayOfWeek <=8"
        area = (
            "StartLatitude < 38.35 & StartLatitude > 38.15 & StartLongitude < -85.55 & StartLongitude > -85.9 "
            "& EndLatitude < 38.35 & EndLatitude > 38.15 & EndLongitude < -85.55 & EndLongitude > -85.9 "
        )
        query = (
            "TripDuration < 60 & TripDuration > 0 & HourNum <= " + str(hour_max) + ""
            "&" + area + " & " + period
        )
        trips_data = trips_data.query(query)

        weather_data = weather_data.query(period + "& Holiday == 0")

        # take out all weather data that corresponds to a weekend. 1.0 is a sunday and 7.0 is a saturday in the dataset
        weekdays = [2, 3, 4, 5, 6]
        weather_data = weather_data.query("DayOfWeek in @weekdays")
        # don't need to filter the trips data because we use the weather data to look-up what trips data to use

        self.weather_data = weather_data
        # get the coordinates of the depots
        _, _, _, _, _, _, centroid_coords = pickle.load(
            open("scripts/bikes_data/training_data_" + "5" + "_" + "40" + ".pckl", "rb")
        )
        R = len(centroid_coords)

        self.centroid_coords = centroid_coords

        self.sim = Rentals_Simulator(
            trips_data, centroid_coords, depth=depth, walk_dist_max=walk_distance_max
        )

        # size of the inputs not controlled by the truck (weather, demand)
        self.z_shape = 3
        if self.chunk_demand:
            self.z_shape = self.z_shape + self.depth + 1

        self.z_max = None

        self.z_max = self.get_z_max()

    def get_env_profile(self):
        """
        Outputs a dictionary containing all details of the environment needed for 
        BO experiments. We override this from the base class to append some bikes-specific properties. 
        """
        # bikes_quantities are properties specific to the bikes to be appended to the env_profile dict. include the centroids
        bikes_quantities = {
            "depth": self.depth,
            "centroids": self.centroids,
            "n_bikes": self.n_bikes,
            "bikes_per_truck": self.bikes_per_truck,
            "z_shape": self.z_shape,
            "trucks": self.N,
            "walk_distance_max": self.sim.walk_dist_max,
            "z_max": self.z_max,
            "centroid_coords": self.sim.centroids,
        }

        return {
            **self.get_base_profile(),
            **self.get_causal_quantities(),
            **bikes_quantities,
        }

    def get_parent_nodes(self):
        """
        Computes a list of lists of the parent nodes for every node. Uses get_parets_row as a helper function to construct this list
        for nodes at each depth. Implemented in sublasses. 
        """
        raise NotImplementedError

    def adapt_active_input_indices(self):
        """
        Computes for every node which input actitions are parents.
        full_input_indices includes actions not by the trucks. 
        Implemented in subclasses. 
        """
        raise NotImplementedError

    def get_z_max(self):

        # just run get_z_t for every t and take the max
        z_max = np.zeros(self.z_shape)
        for t in range(len(self.weather_data)):
            z_t = self.get_z_t(t)
            z_max = np.maximum(z_max, z_t)
        return z_max

    def get_demand_max(self):
        """
        Returns just the max demand for a day. Used for normalizing rewards.
        """
        return self.z_max[2]

    def get_z_t(self, t):
        """
        Given a time t generates the z actions (adversary actions)
        """
        # if t is greater than weather data length throw an error
        assert t < len(self.weather_data), "t is greater than weather data length"

        day_t = np.mod(t, len(self.weather_data))
        daynum = self.weather_data.iloc[day_t].Day
        month = self.weather_data.iloc[day_t].Month
        dayofweek = self.weather_data.iloc[day_t].DayOfWeek

        query = "Day == " + str(daynum) + "& Month == " + str(month)
        weather = self.weather_data.query(query)
        demand = len(self.sim.trips_data.query(query))
        # I also want to get the demand at the first, second third etc self.depth chunk of the day
        # so I will divide the day into self.depth chunks and get the demand in each chunk

        start_times = self.sim.trips_data.query(query).StartTime
        # convert this series from string time format to float in number of hours
        start_times = start_times.apply(
            lambda x: float(x.split(":")[0]) + float(x.split(":")[1]) / 60
        )

        demand_i = []
        # count elements in start_times less than 24/self.depth, 2*24/self.depth, 3*24/self.depth etc
        for i in range(self.depth + 1):
            if i == 0:
                demand_i.append(len(start_times[start_times <= 24 / (self.depth + 1)]))
            else:
                demand_i.append(
                    len(
                        start_times[
                            (start_times > i * 24 / (self.depth + 1))
                            & (start_times <= (i + 1) * 24 / (self.depth + 1))
                        ]
                    )
                )

        if self.chunk_demand:
            z_t = np.concatenate([weather.Temp_Avg, weather.Precip, [demand], demand_i])
        else:
            z_t = np.concatenate([weather.Temp_Avg, weather.Precip, [demand]])

        assert z_t.shape[0] == self.z_shape, "z_t shape is not correct"

        if self.z_max is None:
            return z_t

        # asseert shape the same
        assert z_t.shape == self.z_max.shape, "z_t shape is not correct"
        return z_t / self.z_max

    def evaluate(self, X, X_a, t):
        """
        For a given t, bike assignment X, and adversary action X_a, compute the rewards and intermediate values in the graph. 
        """
        raise NotImplementedError


class BikesSparse(Bikes):
    """
    A sparse version of the bikes environment where we consider only depth 1 and the number of trips in each region only depends on bikes 
    placed at depots within that region. This information is embedded in the parent nodes.
    """

    def __init__(
        self,
        depth=1,
        centroids=116,
        N=5,
        chunk_demand=False,
        alpha=0.01,
        split_reward_by_centroid=True,
        walk_distance_max=1.0,
        norm_reward=False,
    ):
        # Depth basically doesn't matter since we only use depth = 1, but the parent class can have depth > 1.
        self.norm_reward = norm_reward
        self.num_clusters = 15
        self.cluster_labels, self.cluster_centers = pickle.load(
            open(
                "scripts/bikes_data/clustered_centroids_"
                + str(centroids)
                + "_"
                + str(self.num_clusters)
                + ".pckl",
                "rb",
            )
        )

        super().__init__(
            depth,
            centroids,
            N,
            chunk_demand,
            alpha,
            split_reward_by_centroid,
            walk_distance_max,
        )

    def get_parent_nodes(self):
        """
        A list for each node of the indices of other nodes that are parents of that node. 
        For sparse graph all nodes except the reward have no parents besides the inputs. 
        """
        x = [[] for i in range(self.num_clusters)]
        x.append([i for i in range(self.num_clusters)])
        return x

    def adapt_active_input_indices(self):
        """
        active_input_indices is a list for each node of the input indices that are parents of that node.
        full_input_indices adds any non-player inputs to these lists. 
        """
        active_input_indices = []
        full_input_indices = []

        for i in range(self.num_clusters):
            # whether an input causes a node is determined by whether they the centroid corresponding to that node is in the same region/cluster
            active_input_indices.append(
                [j for j in range(self.centroids) if self.cluster_labels[j] == i]
            )
            full_input_indices.append(
                [j for j in range(self.centroids) if self.cluster_labels[j] == i]
            )
            # add to full_input the adversary values (weather and demand)
            full_input_indices[i] = full_input_indices[i] + [
                j for j in range(self.centroids, self.centroids + 3)
            ]  # weather data

        active_input_indices.append([])
        full_input_indices.append(
            [j for j in range(self.centroids, self.centroids + 3)]
        )

        return active_input_indices, full_input_indices

    def evaluate(self, X, X_a, t):
        return self.evaluate_with_trip_coords(X, X_a, t)[0]

    def evaluate_with_trip_coords(self, X, X_a, t):
        # if t is greater than weather data length throw an error
        assert t < len(self.weather_data), "t is greater than weather data length"

        day_t = np.mod(t, len(self.weather_data))
        daynum = self.weather_data.iloc[day_t].Day
        month = self.weather_data.iloc[day_t].Month
        dayofweek = self.weather_data.iloc[day_t].DayOfWeek

        query = "Day == " + str(daynum) + "& Month == " + str(month)
        weather = self.weather_data.query(query)
        demand = len(self.sim.trips_data.query(query))

        z_t = self.get_z_t(t)

        # Transform X to an np.array with the number of bikes at each centroid
        X_sim_input = np.zeros(len(self.sim.centroids))
        for i in X:
            X_sim_input[i] += self.bikes_per_truck

        trips_starting_coords, trips_unmet_coords, x_new_chunks = self.sim.simulate_rentals(
            X_sim_input, daynum, month
        )

        # Go through trips_starting_coords, find the closest cluster to each (euclidean distance) and count them towards that cluster

        trips_per_cluster = np.zeros(self.num_clusters)
        for trip in trips_starting_coords:
            # find the closest cluster
            closest_cluster = 0
            closest_centroid = 0
            closest_d = 1000000
            for i in range(self.centroids):
                d = np.linalg.norm(trip - self.centroid_coords[i])
                if d < closest_d:
                    closest_d = d
                    closest_centroid = i
            closest_cluster = self.cluster_labels[closest_centroid]
            trips_per_cluster[closest_cluster] += 1

        trips_per_cluster = trips_per_cluster / self.get_demand_max()

        reward = np.sum(trips_per_cluster)
        output = (
            np.concatenate([trips_per_cluster.flatten(), reward.flatten()]),
            trips_starting_coords,
            trips_unmet_coords,
            x_new_chunks,
        )

        return output

    def get_max_per_node(self):
        """
        Calls evaluate with a packed X (bikes everywhere) for every t then takes a max to get the max per node
        """
        max_per_node = np.zeros(self.num_clusters + 1)
        for t in range(len(self.weather_data)):
            round_t_eval = self.evaluate(5 * [i for i in range(self.centroids)], 1, t)

            max_per_node = np.maximum(max_per_node, round_t_eval)
        return max_per_node
