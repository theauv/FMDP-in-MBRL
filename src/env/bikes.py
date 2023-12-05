"""
Environments for the bikes experiments.
I would first recommend looking at functions.py since the class structure is similar 
and the environments are much simpler. 
"""

from typing import Optional, Dict, Tuple

import geopy.distance
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces.utils import flatten, unflatten
import matplotlib
import matplotlib.backends.backend_agg as agg
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import omegaconf
import pandas as pd
import pickle
import pylab
from scipy.spatial import distance
import torch
import warnings

from src.util.util import my_draw_networkx_edge_labels


class Rentals_Simulator:
    """ Class used to simulate rentals in the city from historic data."""

    def __init__(self, trips_data, centroids, walk_dist_max=1):
        self.trips_data = trips_data  # history of trips data (used to simulate rentals)
        self.centroids = (
            centroids
        )  # centroids for regions in the city where bikes/scooters are positioned
        self.R = len(
            centroids
        )  # number of regions in the city TODO: don't like the multiple definitions of R
        # self.chunks = depth +1 #number of time chunks to break the day into. You add 1 to the depth
        self.walk_dist_max = walk_dist_max
        self.taken_bikes = []

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
        tot_demand_per_centroid = np.zeros(R, dtype=int)
        met_trips_per_centroid = np.zeros(R, dtype=int)

        adjacency_matrix = np.zeros((R, R), dtype=int)

        # TODO: maybe not a fixed time for every trip ?
        TRIP_DURATION = (
            0.5
        )  # in hours: the time a bike is removed from the system for while a trip is happening

        #taken_bikes = []
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
            for bike in self.taken_bikes:
                if bike[0] <= start_time:
                    new_x[bike[1]] += 1
                    self.taken_bikes.remove(bike)

            distances = distance.cdist(
                start_loc.reshape(-1, 2), self.centroids, metric="euclidean"
            )
            idx_argsort = np.argsort(distances)[0]
            # We go through the centroids in order of closest to farthest and see if there is an available bike at one of the centroids to make the trip
            for i, idx_start_centroid in enumerate(idx_argsort):
                if (
                    geopy.distance.distance(
                        start_loc, self.centroids[idx_start_centroid, :]
                    ).km
                    > self.walk_dist_max
                ):
                    if i == 0:
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
                    self.taken_bikes.append((start_time + TRIP_DURATION, idx_end_centroid))

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


class Bikes(gym.Env):
    # WHAT IS METADATA ???
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(
        self,
        env_config: Optional[omegaconf.DictConfig],
        render_mode: Optional[str] = None,
    ) -> None:

        # TODO: self.split_reward_by_centroid = env_config.split_reward_by_centroid # if true, we model the number of trips from each centroid individual, instead of the total number of trips
        # TODO: not especially optimal, could also decide for fix number of actions WHEN to take them during the day ??
        # TODO: "self strips" ?? Should we consider them or not (only influence is the trip_duration)
        self.num_trucks = env_config.num_trucks
        self.action_per_day = env_config.action_per_day
        self.day_start = 0.0
        self.day_end = 24.0
        self.action_timeshifts = list(
            np.linspace(self.day_start, self.day_end, self.action_per_day + 1)
        )
        self.sample_method = env_config.sample_method
        self.initial_distribution = env_config.initial_distribution

        # TODO: rewrite this in a modulable way
        base_dir = env_config.base_dir
        self.centroid_trips_matrix = pd.read_pickle(
            open(base_dir+"src/env/bikes_data/centroid_trips_matrix5_40.pckl", "rb")
        )
        _, _, _, _, _, _, centroid_coords = pickle.load(
            open(base_dir+"src/env/bikes_data/training_data_5_40.pckl", "rb")
        )
        R = len(centroid_coords)
        self.num_centroids = R

        self.centroid_coords = centroid_coords
        self.bikes_per_truck = (
            env_config.bikes_per_truck
        )
        self.n_bikes = self.num_trucks * self.bikes_per_truck

        # TODO: normalize and how do we do with discrete values ???
        self.observation_space = spaces.Dict(
            {   
                "bikes_dist_before_shift": spaces.MultiDiscrete(
                    np.ones(self.num_centroids) * self.n_bikes,
                ),
                "bikes_dist_after_shift": spaces.MultiDiscrete(
                    np.ones(self.num_centroids) * self.n_bikes,
                ),
                "day": spaces.Discrete(31, start=1),
                "month": spaces.Discrete(12, start=1),
                "time_counter": spaces.Discrete(self.action_per_day+1),
            }
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
            base_dir+"src/env/bikes_data/dockless-vehicles-3_full.csv",
            usecols=lambda x: x not in ["TripID", "StartDate", "EndDate", "EndTime"],
        )
        self.all_weather_data = pd.read_csv(
            base_dir+"src/env/bikes_data/weather_data.csv",
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

    def get_timeshift(self, state=None):
        if state is None:
            state = self.state
        counter = state["time_counter"]
        if counter < len(self.action_timeshifts)-1:
            return self.action_timeshifts[counter:counter+2]
        elif counter == len(self.action_timeshifts)-1:
            return None
        else:
            raise ValueError("Episode should be terminated")

    def trips_steps(self, x=None):

        if x is None:
            x = self.state

        mask = (self.all_trips_data["Day"] == x["day"]) & (
            self.all_trips_data["Month"] == x["month"]
        )
        current_trips = self.all_trips_data[mask]

        # Keep only the trip at times we care:
        timeshift = self.get_timeshift(x)
        start_time = timeshift[0]
        end_time = timeshift[1]

        time_mask = [
            float(time.replace(":", ".").split(":")[0]) >= start_time
            and float(time.replace(":", ".").split(":")[0]) <= end_time
            for time in current_trips["StartTime"].values
        ]

        current_trips = current_trips[time_mask]

        # Compute the new state and all relevant informations about trips occuring during this timeshift
        (
            new_bikes_dist_after_shift,
            self.tot_num_trips,
            self.num_met_trips,
            self.too_far_from_centroid_trips,
            trips_starting_coords,
            met_trips_starting_coords,
            too_far_from_centroid_trips_coords,
            self.tot_demand_per_centroid,
            self.met_trips_per_centroid,
            self.adjacency,
        ) = self.sim.simulate_rentals(current_trips, x["bikes_dist_before_shift"])

        reward = self.compute_reward()

        return new_bikes_dist_after_shift, reward        

    def compute_reward(self) -> float:
        # TODO: Ideas
        # trips_met/(tot_trips-too_far_from_centroid)
        # mean(met_demand_percentroid/tot_demand_per_centroid)
        # add the number of added bikes as a penalty
        # add the gasoil comsuption for each bike refill
        # warning if too_far_from_centroid > 0

        # Remark, adding the possibility to get add or remove bikes is somewhat equivalent to rebalancing

        if self.too_far_from_centroid_trips>0:
            warnings.warn(
                f"We have {self.too_far_from_centroid_trips} trips that could not met because there was no centroid close enough"
            )

        relevant_trips = max(1, self.tot_num_trips - self.too_far_from_centroid_trips)
        centroid_ratio = np.where(self.tot_demand_per_centroid != 0, self.met_trips_per_centroid / self.tot_demand_per_centroid, np.nan)
        centroid_ratio = np.nanmean(centroid_ratio)

        return self.num_met_trips / relevant_trips + centroid_ratio

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:

        if type(action)==np.ndarray:
            action = unflatten(self.action_space, action)

        # TODO: model could be a GNN

        # Update state: state = f(old_state, action)
        old_state = self.state
        self.delta_bikes = np.zeros(self.num_centroids, dtype=int)
        truck_centroid = action["truck_centroid"]
        truck_num_bikes = action["truck_num_bikes"]
        for truck in range(self.num_trucks):
            self.delta_bikes[truck_centroid[truck]] += truck_num_bikes[truck]

        self.state["bikes_dist_before_shift"] = self.state["bikes_dist_after_shift"] + self.delta_bikes

        # Let all the vehicules being used during the day
        new_bikes_dist_after_shift, reward = self.trips_steps()
        self.state["bikes_dist_after_shift"] = new_bikes_dist_after_shift

        # Render the environment
        if self.render_mode == "human":
            self.render()

        #Step the time counter
        self.state["time_counter"] += 1

        #Check if terminated
        terminated = (self.get_timeshift() is None)

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
            flatten(self.observation_space, self.state),
            reward,
            terminated,
            False,
            {},
        )  # observation, reward, end, truncated, info

    def get_initial_bikes_distribution(self) -> np.array:
        if self.initial_distribution == "uniform":
            # bikes_per_region = self.n_bikes // self.num_centroids
            # x = np.ones(self.num_centroids, dtype=int) * bikes_per_region
            x = np.zeros(self.num_centroids, dtype=int)
            for i, bike in enumerate(range(self.n_bikes)):
                x[i%self.num_centroids] += 1
        elif self.initial_distribution == "zeros":
            x = np.zeros(self.num_centroids, dtype=int)
        else:
            raise ValueError(
                f"There is no such initial bike distribution called {self.initial_distribution}"
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
            ].iloc[0]
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
            "bikes_dist_before_shift": self.get_initial_bikes_distribution(),
            "bikes_dist_after_shift": self.get_initial_bikes_distribution(),
            "day": day,
            "month": month,
            "time_counter": 0,
        }

        self.tot_num_trips = None
        self.num_met_trips = None
        self.too_far_from_centroid_trips = None
        self.tot_demand_per_centroid = None
        self.met_trips_per_centroid = None
        self.adjacency = None
        self.delta_bikes = None

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
        #self.surf.fill(BLACK)

        #Plots
        matplotlib.use("Agg")
        fig, axs = pylab.subplots(
            1,
            2,
            figsize=[7, 3.5],  # Inches
            dpi=200,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
        )
        # ax = fig.gca()
        if self.delta_bikes is not None:
            G_1 = nx.DiGraph()

            for i in range(self.num_centroids):
                # node_label = f"{self.state['bikes_dist_after_shift'][i]} \n (+{self.delta_bikes[i]})"
                node_label = f"+{self.delta_bikes[i]}" if self.delta_bikes[i]>0 else ""
                node_label += f"\n ({self.met_trips_per_centroid[i]}/{self.tot_demand_per_centroid[i]})" if self.tot_demand_per_centroid[i]>0 else ""
                G_1.add_node(i, pos=self.centroid_coords[i], color="green", weight=self.state['bikes_dist_after_shift'][i], label=node_label)

            pos = nx.get_node_attributes(G_1, "pos")
            node_colors = nx.get_node_attributes(G_1, "color")
            node_weights = nx.get_node_attributes(G_1, "weight")
            node_labels = nx.get_node_attributes(G_1, "label")
            nx.draw_networkx(G_1, with_labels=True, pos=pos, node_color=node_colors.values(), node_size=[v/2 + 1 for v in node_weights.values()], font_size=3, ax=axs[0], labels=node_labels)

            G_2 = nx.DiGraph()

            for i in range(self.num_centroids):
                #node_label = f"{self.state['bikes_dist_after_shift'][i]} \n ({self.met_trips_per_centroid[i]}/{self.tot_demand_per_centroid[i]})"
                node_label = f"{self.met_trips_per_centroid[i]}/{self.tot_demand_per_centroid[i]}"
                G_2.add_node(i, pos=self.centroid_coords[i], color="green", weight=self.state['bikes_dist_after_shift'][i], label=node_label)

            for i, centroid_i in enumerate(self.adjacency):
                for j, centroid_j in enumerate(centroid_i):
                    if centroid_j > 0:
                        G_2.add_edge(i, j, weight=centroid_j)

            #Set up
            pos = nx.get_node_attributes(G_2, "pos")
            node_colors = nx.get_node_attributes(G_2, "color")
            node_weights = nx.get_node_attributes(G_2, "weight")
            node_labels = nx.get_node_attributes(G_2, "label")

            #Nodes
            node_size = (self.n_bikes // self.num_centroids)/4 + 1
            nx.draw_networkx_nodes(G_2, pos, node_color=node_colors.values(), node_size=node_size, ax=axs[1]) #node_size=[v for v in node_weights.values()], ax=axs[1])
            #nx.draw_networkx_labels(G_2, pos, labels=node_labels, font_size=10, ax=axs[1])

            #Edges (curved/straight)
            curved_edges = [edge for edge in G_2.edges() if reversed(edge) in G_2.edges()]
            straight_edges = list(set(G_2.edges()) - set(curved_edges))
            nx.draw_networkx_edges(G_2, pos, edgelist=straight_edges, arrowsize=1, width=0.5, node_size=node_size, ax=axs[1])
            arc_rad = 0.1
            nx.draw_networkx_edges(G_2, pos, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}',  arrowsize=1, width=0.5, node_size=node_size, ax=axs[1])

            #Label edges
            # edge_weights = nx.get_edge_attributes(G_2,'weight')
            # curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
            # straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
            # my_draw_networkx_edge_labels(G_2, pos, edge_labels=curved_edge_labels, rotate=False, rad = arc_rad, font_size=1, ax=axs[1])
            # nx.draw_networkx_edge_labels(G_2, pos, edge_labels=straight_edge_labels,rotate=False, font_size=1, ax=axs[1])

        else:
            G_1 = nx.DiGraph()

            for i in range(self.num_centroids):
                node_label = f"{self.state['bikes_dist_before_shift'][i]}"
                G_1.add_node(i, pos=self.centroid_coords[i], color="green", weight=self.state['bikes_dist_before_shift'][i], label=node_label)

            pos = nx.get_node_attributes(G_1, "pos")
            node_colors = nx.get_node_attributes(G_1, "color")
            node_weights = nx.get_node_attributes(G_1, "weight")
            node_labels = nx.get_node_attributes(G_1, "label")
            nx.draw_networkx(G_1, with_labels=True, pos=pos, node_color=node_colors.values(), node_size=[v/2 + 1 for v in node_weights.values()], labels=node_labels, font_size=3, ax=axs[0])
            #nx.draw_networkx(G_1, with_labels=False, pos=pos, node_color=node_colors.values(), node_size=[v/2 for v in node_weights.values()], font_size=3, ax=axs[1])

        title_1 = f'Month: {self.state["month"]} Day: {self.state["day"]} Timeshift: {self.get_timeshift()}'
        if self.num_met_trips is not None:
            relevant_trips = max(1, self.tot_num_trips - self.too_far_from_centroid_trips)
            ratio_met_trips = round(self.num_met_trips/relevant_trips, 2)
            centroid_ratio = np.where(self.tot_demand_per_centroid != 0, self.met_trips_per_centroid / self.tot_demand_per_centroid, np.nan)
            centroid_ratio = round(np.nanmean(centroid_ratio), 2)
            title_2 = f'Ratio met trips: {ratio_met_trips} Ratio met demands: {centroid_ratio} Non-relevant trips: {self.too_far_from_centroid_trips}'
            title = title_1+'\n'+title_2
        else:
            title = title_1

        fig.suptitle(title, fontsize=10)
        fig.canvas.manager.full_screen_toggle()
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
        #new_all_obs = []
        done = []
        for obs in next_obs:
            obs = obs.numpy()
            if type(obs) == np.ndarray:
                obs = unflatten(self.observation_space, obs)
            done.append(self.get_timeshift(obs) is None)

        return done  

    def reward_fn(self, actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """
        Reward function associated to the hypergrid env

        :param action: batch of actions
        :param next_obs: batch of next_obs
        :return: batch of rewards associated to each (action, next_obs)
        """
        unflatten_next_obs = []
        for obs in next_obs:
            obs = obs.numpy()
            if type(obs) == np.ndarray:
                obs = unflatten(self.observation_space, obs)
            unflatten_next_obs.append(obs)

            #Modify the observation to recompute the reward
            obs["time_counter"] -= 1
            obs["bikes_dist_after_shift"] = obs["bikes_dist_before_shift"]

        rewards = []
        for obs in unflatten_next_obs:
            _, reward = self.trips_steps(obs)
            rewards.append(reward)

        return rewards