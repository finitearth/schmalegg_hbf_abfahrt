import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from gym.spaces import box

import generate_envs
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
                if s1 != s2:
                    edge_index[0].append(s1)
                    edge_index[1].append(s2)
        #edge_index = [range(5), range(5)]
        self.edge_index = edge_index
        self.reachable_stations = [0, 1, 2, 3, 4]
        self.i = 0

    def step(self, action):
        self.i += 1
        #print(action)
        reward = 0
        done = False
        info = {"huhu kann man lesen?": "Nein"}
        train_vector = np.array([1, 1, 1, 1])
        #print(type(action))
        stop_vectors = [np.array(action[i:i + 4]) for i in range(0, action.size, 4)]
       # print(stop_vectors[2])
        #print(stop_vectors[0])

        max_prod = -1000
        best_stop = -1  # self.train_pos
        for station in self.reachable_stations:

            dot_prod = np.dot(stop_vectors[station], train_vector)
           # print(f"Station: {station}: Prod: {dot_prod}")
            if dot_prod > max_prod and station != self.train_pos:
                #print("isch größa")
                max_prod = dot_prod
                best_stop = station
        self.train_pos = best_stop
        assert 0 <= self.train_pos <= 4, "Train has no valid position"

        if self.train_pos == 0:
          #  print("eingstige", self.i)
            self.trains[0].passengers.append(self.passengers[0])
            self.stations[0].passengers = [0, 0, 0, 0, 0]
            #reward += 5


        if self.train_pos == 2 and self.trains[0].passengers != []:
            #print("ausgstige", self.i)
            #reward += 10
            done = True

        if not done:  # passenger.reached_destination():
            reward -= 100

      #  else: print("ziel erreicht", self.i)

        observation = self.get_observation()
        return observation, reward, done, info

    def reset(self):
        self.stations, self.passengers, self.trains = generate_envs.generate_random_env()

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
