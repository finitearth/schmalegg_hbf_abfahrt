import math
import gym
import torch
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from gym.spaces import box
from torch_geometric.data import Data, HeteroData

import generate_envs
import objects
import random

OUTPUT_VECTOR_SIZE = 2


class AbfahrtEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super(AbfahrtEnv, self).__init__()
        self.stations = []
        self.trains = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.score = 0
        self.routes = np.zeros((100, 100))
        # self.keys = "str"
        self.k = 0

    def step(self, action):
        self.k += 1
        info = {}
        action = action[:OUTPUT_VECTOR_SIZE * len(self.stations)]

        for i, station in enumerate(self.stations):
            station.vector = action[i * OUTPUT_VECTOR_SIZE:(i + 1) * OUTPUT_VECTOR_SIZE]

        for train in self.trains:
            # check if reached destination; if not: skip
            if not train.reached_next_stop(): continue

            # deboard and onboard passengers
            for p in train.passengers: train.deboard(p)

            while len(train.passengers) < train.capacity and len(train.station.passengers) != 0:
                dot_products = np.asarray(
                    [train.station.vector @ p.destination.vector for p in train.station.passengers])
                idx = np.argmax(dot_products)
                if dot_products[idx] > 0:
                    train.onboard(train.station.passengers[idx])
                else: break

            # route to the next stop
            if len(train.passengers) > 0:
                train_vector = np.sum([p.destination.vector for p in train.passengers], axis=0)
            else:
                train_vector = np.asarray([1] * OUTPUT_VECTOR_SIZE)

            next_stop_idx = np.argmax([train_vector.T @ s.vector for s in train.station.reachable_stops])
            train.reroute_to(train.station.reachable_stops[next_stop_idx])

        score = np.sum([len(s.passengers) for s in self.stations])
        score += np.sum([len(t.passengers) for t in self.trains])
        reward = (self.score - score)*5 - 1
        self.score = score
        # reward = -1.0

        done = bool((np.sum([len(s.passengers) for s in self.stations]) + np.sum(
            [len(t.passengers) for t in self.trains])) == 0)

        # if done: reward = +5.0
        observation = self.get_observation()
        return observation, reward, done, info

    def reset(self, mode="train"):
        self.score = np.sum([len(s.passengers) for s in self.stations])
        if mode == "train":
            # Generate random enviroment for training
            self.routes, self.stations, self.trains = generate_envs.generate_random_env()#generate_envs.generate_example_enviroment()#
        elif mode == "eval":
            # Generate evaluation enviroment
            self.routes, self.stations, self.trains = generate_envs.generate_random_env()#
        return self.get_observation()

    def render(self, mode="human"):
        raise NotImplementedError

    def get_observation(self):
        n_stations = len(self.stations)

        edge_index_connections0 = self.routes[0]
        n_edge_connections = len(edge_index_connections0)
        edge_index_connections0 = np.resize(edge_index_connections0, 100)

        edge_index_connections1 = self.routes[1]
        edge_index_connections1 = np.resize(edge_index_connections1, 100)

        edge_index_destination0 = []
        edge_index_destination1 = []
        n_passenger = 0
        for s in self.stations:
            for p in s.passengers:
                edge_index_destination0 += [int(s)]
                edge_index_destination1 += [int(p.destination)]
                n_passenger += 1
        for t in self.trains:
            for p in t.passengers:
                edge_index_destination0 += [int(t.destination)]
                edge_index_destination1 += [int(p.destination)]
                n_passenger += 1
        edge_index_destination0 = np.asarray(edge_index_destination0)
        edge_index_destination0 = np.resize(edge_index_destination0, 1000)
        edge_index_destination1 = np.asarray(edge_index_destination1)
        edge_index_destination1 = np.resize(edge_index_destination1, 1000)

        input_vectors = np.hstack([s.getencoding() for s in self.stations])
        input_vectors = torch.tensor(np.resize(input_vectors, 1000))

        return np.hstack((
            edge_index_connections0,
            edge_index_connections1,
            edge_index_destination1,  # ###weil die information von zielbahnhof zu aktuellem bahnhof fließen muss
            edge_index_destination0,  #################
            input_vectors,
            n_passenger,
            n_edge_connections,
            n_stations
        ))
        # return CustomData(x=input_vectors, edge_index_destinations=[edge_index_destination0, edge_index_destination1],
        #                   edge_index_connections=self.routes)

    def close(self):
        pass
