import glob
import gym
import networkx
import torch
import numpy as np
from gym.spaces import Box
import objects
from objects import EnvBlueprint


class AbfahrtEnv(gym.Env):
    def __init__(self, config, mode="eval"):
        super(AbfahrtEnv, self).__init__()
        self.stations = []
        self.trains = []
        self.observation_space = Box(-100, +100, shape=(150_003,), dtype=np.float32)
        self.action_space = Box(-100, +100, shape=(50_000,), dtype=np.float32)
        self.routes = None
        self.shortest_path_lenghts = None
        self.action_vector_size = config.action_vector_size
        self.n_node_features = config.n_node_features
        self.config = config
        self.active_passengers = 0
        self.min_steps_to_go = 0

        self.mode = mode

        self.train_envs = []
        self.eval_envs = []
        for file in glob.glob("./graphs/eval/*"):
            env = EnvBlueprint()
            env.read(file)
            self.eval_envs.append(env)
        self.resets = 0
        self.step_count = 0 # help me stepcount, im stuck!

    def step(self, action):
        self.step_count += 1
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy()
        action = action[:self.action_vector_size * len(self.stations)]
        for i, station in enumerate(self.stations):
            station.vector = action[i * self.action_vector_size:(i + 1) * self.action_vector_size]

        for train in self.trains: # rerouting of trains and onboarding
            if not train.reached_next_stop(): continue
            for p in train.passengers: train.deboard(p)
            while len(train.passengers) < train.capacity and len(train.station.passengers) != 0:
                dot_products = train.station.vector @ [p.destination.vector for p in train.station.passengers]
                idx = np.argmax(dot_products)

                if dot_products[idx] > 0: train.onboard(train.station.passengers[idx])
                else: break

            train_vector = train.station.vector
            next_stop_idx = np.argmax(train_vector @ [s.vector for s in train.station.reachable_stops])
            train.reroute_to(train.station.reachable_stops[next_stop_idx])

        min_steps_to_go = self.config.reward_step_closer * self._min_steps_to_go()
        reward = self.config.reward_step_closer * (self.min_steps_to_go - min_steps_to_go)
        self.min_steps_to_go = min_steps_to_go

        active_passengers = sum([len(s.passengers) for s in self.stations+self.trains])
        reward += (self.active_passengers-active_passengers)*self.config.reward_reached_dest+self.config.reward_per_step
        self.active_passengers = active_passengers

        done = bool(active_passengers == 0)
        if self.step_count > 500: # stop after 500 steps, because aint nobody got time for that
            done = True
            reward = -100

        return self.get_observation(), reward, done, {}

    def _min_steps_to_go(self):
        sd = []
        for st in self.trains + self.stations:
            s = st.destination if isinstance(st, objects.Train) else st
            for p in st.passengers:
                sd.append((int(s), int(p.destination)))

        c = sum([self.shortest_path_lenghts[s][d] for s, d in sd])
        return c

    def reset(self):
        # files = glob.glob("../graphs/eval/*")
        if self.resets % self.config.batch_size == 0:
            for _ in range(self.config.batch_size):
                env = EnvBlueprint()
                env.random(n_max_stations=30)
                self.train_envs.append(env)

        if self.mode == "train":
            n_envs = len(self.train_envs) - 1
            env_idx = int(min(1, abs(np.random.normal(loc=0, scale=0.5)))*n_envs)
            self.routes, self.stations, self.trains = self.train_envs[env_idx].get()
        elif self.mode == "eval":
            self.routes, self.stations, self.trains = self.eval_envs[self.resets % len(self.eval_envs)].get()
            self.resets += 1

        elif self.mode == "render":
            self.routes, self.stations, self.trains = self.eval_envs[0].get()

        for s in self.stations: s.set_input_vector(n_node_features=self.config.n_node_features)
        edges = self.routes
        edges = list(zip(edges[0], edges[1]))
        g = networkx.Graph()
        for edge in edges:
            g.add_edge(edge[0], edge[1])
        self.shortest_path_lenghts = dict(networkx.shortest_path_length(g))
        self.min_steps_to_go = self._min_steps_to_go()
        self.active_passengers = sum([len(s.passengers) for s in self.stations])
        self.step_count = 0
        return self.get_observation()

    def get_observation(self):
        n_stations = len(self.stations)

        edge_index_connections0 = self.routes[0]
        n_edge_connections = len(edge_index_connections0)
        edge_index_connections0 = np.resize(edge_index_connections0, 25_000)

        edge_index_connections1 = self.routes[1]
        edge_index_connections1 = np.resize(edge_index_connections1, 25_000)

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
        edge_index_destination0 = np.resize(edge_index_destination0, 25_000)
        edge_index_destination1 = np.asarray(edge_index_destination1)
        edge_index_destination1 = np.resize(edge_index_destination1, 25_000)

        input_vectors = np.hstack([s.getencoding() for s in self.stations])
        input_vectors = np.resize(input_vectors, 50_000)

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
