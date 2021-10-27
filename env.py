import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from gym.spaces import box

import generate_random
import objects


class AbfahrtEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super(AbfahrtEnv, self).__init__()
        self.passengers = []
        self.stations = []
        self.trains = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.train_pos = 0
        edge_index = [[], []]
        for s1 in range(5):
            for s2 in range(5):
                edge_index[0].append(s1)
                edge_index[1].append(s2)
        self.edge_index = edge_index

    def step(self, action):
        reward = 0
        done = False
        info = {"huhu kann man lesen?": "Nein"}
        train_vector = np.array([1, 1, 1, 1])
        stop_vectors = [np.array(action[i:i + 4]) for i in range(0, action.size, 4)]
        # print(action.size)
        reachable_stations = []
        for i in range(len(self.stations)):
            if self.edge_index[0][i] == self.train_pos: reachable_stations.append(self.edge_index[1][i])
        max_prod = float("-inf")
        best_stop = -1
        for station in reachable_stations:
            dot_prod = np.dot((stop_vectors[station]), train_vector)
            if dot_prod > max_prod:
                max_prod = dot_prod
                best_stop = station
        self.train_pos = best_stop

        for passenger in self.passengers:
            if 0 == self.train_pos: self.trains[0].passengers.append(passenger); self.stations[0].passengers = [0, 0, 0, 0, 0]
            if self.train_pos == 2 and self.trains[0].passengers != []: done = True

        if not done:  # passenger.reached_destination():
            reward -= 1

        observation = self.get_observation()
        return observation, reward, done, info

    def reset(self):
        self.stations, self.passengers, self.trains = generate_random.generate_random_env()

        return self.get_observation()

    def render(self, mode="human"):
        raise NotImplementedError

    def get_observation(self):
        observation = []
        for station in self.stations:
            observation.append(station.getencoding())
        observation = np.asarray(observation, dtype=np.float32).flatten()
        self.observation = observation

        return observation

    def close(self):
        pass
