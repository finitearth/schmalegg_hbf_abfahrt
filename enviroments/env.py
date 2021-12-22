import time
import gym
import torch
import numpy as np
from gym.spaces import Box
from enviroments import generate_envs


class AbfahrtEnv(gym.Env):
    def __init__(self, config):
        super(AbfahrtEnv, self).__init__()
        self.stations = []
        self.trains = []
        self.observation_space = Box(-100, +100, shape=(3203,), dtype=np.float32)
        self.action_space = Box(-100, +100, shape=(400,), dtype=np.float32)
        self.routes = np.zeros((100, 100))
        self.action_vector_size = config.action_vector_size
        self.n_node_features = config.n_node_features
        self.config = config
        self.active_passengers = 0

    def step(self, action):
        action = action[:self.action_vector_size * len(self.stations)]
        for i, station in enumerate(self.stations):
            station.vector = action[i * self.action_vector_size:(i + 1) * self.action_vector_size]

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
            train_vector = train.station.vector
            # if len(train.passengers) > 0:
            #     train_vector = np.sum([p.destination.vector for p in train.passengers], axis=0)
            # else:

            next_stop_idx = np.argmax([train_vector @ s.vector for s in train.station.reachable_stops])
            train.reroute_to(train.station.reachable_stops[next_stop_idx])

        # active_passengers = np.sum([len(s.passengers) for s in self.stations+self.trains])
        active_passengers = np.sum([len(s.passengers) for s in self.stations])
        active_passengers += np.sum([len(t.passengers) for t in self.trains])
        reward = (self.active_passengers - active_passengers) * self.config.reward_reached_dest + self.config.reward_per_step
        self.active_passengers = active_passengers

        done = bool((active_passengers == 0))
        if done: reward += 5
        observation = self.get_observation()
        return observation, reward, done, {}

    def reset(self, mode="train"):
        self.active_passengers = np.sum([len(s.passengers) for s in self.stations])
        if mode == "train":
            # Generate random enviroment for training
            self.routes, self.stations, self.trains = generate_envs.generate_random_env(self.n_node_features)#generate_envs.generate_example_enviroment()#
        elif mode == "eval":
            # Generate evaluation enviroment
            self.routes, self.stations, self.trains = generate_envs.generate_random_env(self.n_node_features)#generate_envs.generate_example_enviroment()#
        return self.get_observation()

    def render(self, mode="human"):
        raise NotImplementedError

    def get_observation(self):
        n_stations = len(self.stations)

        edge_index_connections0 = self.routes[0]
        n_edge_connections = len(edge_index_connections0)
        edge_index_connections0 = np.resize(edge_index_connections0, 1000)

        edge_index_connections1 = self.routes[1]
        edge_index_connections1 = np.resize(edge_index_connections1, 1000)

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
            edge_index_destination0,  #
            input_vectors,
            n_passenger,
            n_edge_connections,
            n_stations
        ))
        #   TODO
        # return CustomData(x=input_vectors, edge_index_destinations=[edge_index_destination0, edge_index_destination1],
        #                edge_index_connections=self.routes)

    def close(self):
        pass
