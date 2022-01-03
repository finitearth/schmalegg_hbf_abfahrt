import glob
import gym
import networkx
import torch
import numpy as np
from dill import dumps
from gym.spaces import Box
import objects
from objects import EnvBlueprint


class AbfahrtEnv(gym.Env):
    def __init__(self, config, mode="eval", using_mcts=False):
        super(AbfahrtEnv, self).__init__()
        self.stations = []
        self.trains = []
        self.trains_dict = {}
        self.stations_dict = {}

        self.observation_space = Box(-100, +100, shape=(200004,), dtype=np.float32)
        self.action_space = Box(-100, +100, shape=(50_000,), dtype=np.float32)
        self.routes = None
        self.shortest_path_lenghts = None
        self.action_vector_size = config.action_vector_size
        self.n_node_features = config.n_node_features
        self.config = config
        self.active_passengers = 0
        self.min_steps_to_go = 0
        self.using_mcts = using_mcts
        self.training = "ppo"

        self.get_ppo_action = None
        self.get_mcts_actino = None
        self.mcts_list = None
        self.mode = mode
        self.inference_env_bp = None
        self.train_envs = []
        self.eval_envs = []
        if self.mode != "inference":
            for file in glob.glob("./graphs/eval/*.json"):
                env = EnvBlueprint()
                env.read_json(file)
                self.eval_envs.append(env)
        self.resets = 0
        self.step_count = 0 # help me stepcount, im stuck!

    def step(self, action):
        if self.training == "ppo":
            ppo_action = action
            mcts_action = None
        elif self.training == "mcts":
            ppo_action = None
            mcts_action = action
        n = self.config.action_vector_size//2
        if ppo_action is None: ppo_action = self.get_ppo_action(self.get_observation())[0][0]
        if mcts_action is None and (self.mcts_list is None or len(self.mcts_list) < 2):
            self.training = "mcts"
            self.mcts_list = self.get_mcts_action(self.get_snapshot(), self.get_observation())
            mcts_action = self.mcts_list[0]
            self.mcts_list.pop(0)
            self.training = "ppo"
        elif mcts_action is None and self.mcts_list is not None:
            mcts_action = self.mcts_list[1]
            self.mcts_list.pop(0)

        self.step_count += 1
        # if isinstance(ppo_action, torch.Tensor):
        #     ppo_action = ppo_action.detach().numpy()
        # ppo_action = ppo_action[:self.action_vector_size * len(self.stations)]
        for i, station in enumerate(self.stations):
            station.vector = ppo_action[i * self.action_vector_size:(i + 1) * self.action_vector_size]

        for train in self.trains: # onboarding
            if not train.reached_next_stop(): continue
            for p in train.passengers: train.deboard(p)
            while len(train.passengers) < train.capacity and len(train.station.passengers) != 0:
                dot_products = [train.station.vector[n:] @ p.destination.vector[:n] for p in train.station.passengers]
                idx = np.argmax(dot_products)

                if dot_products[idx] > 0: train.onboard(train.station.passengers[idx])
                else: break

        self.rerouting_trains(mcts_action)

        min_steps_to_go = self._min_steps_to_go()
        reward = self.config.reward_step_closer * (self.min_steps_to_go - min_steps_to_go)
        self.min_steps_to_go = min_steps_to_go

        active_passengers = sum([len(s.passengers) for s in self.stations+self.trains])
        reward += (self.active_passengers-active_passengers)*self.config.reward_reached_dest+self.config.reward_per_step
        self.active_passengers = active_passengers
        # print(active_passengers)
        done = bool(active_passengers == 0)
        if self.step_count > 1000: # stop after 500 steps, because aint nobody got time for that
            done = True
            reward = -5

        return self.get_observation(), reward, done, {}

    def rerouting_trains(self, mcts_action=None):
        n = self.config.action_vector_size // 2
        if self.using_mcts:
            for (train, station) in mcts_action:
                train = self.trains_dict[int(train)]
                station = self.stations_dict[int(station)]
                train.reroute_to(station)
        else:
            for train in self.trains:
                next_stop_idx = np.argmax([train.station.vector[n:] @ s.vector[:n] for s in train.station.reachable_stops])
                train.reroute_to(train.station.reachable_stops[next_stop_idx])

    def _min_steps_to_go(self):
        c = 0
        for st in self.trains + self.stations:
            s = st.destination if isinstance(st, objects.Train) else st
            for p in st.passengers:
                c += self.shortest_path_lenghts[int(s)][int(p.destination)]
        return c

    def reset(self):
        if self.resets % (self.config.batch_size + len(self.eval_envs)) == 0 and self.mode != "inference":
            for _ in range(self.config.batch_size):
                env = EnvBlueprint()
                env.random(n_max_stations=10)
                self.train_envs.append(env)

        if self.mode == "train":
            n_envs = len(self.train_envs) - 1
            env_idx = int(min(1, abs(np.random.normal(loc=0, scale=5)))*n_envs)
            self.routes, self.stations, self.trains = self.train_envs[env_idx].get()
        elif self.mode == "eval":
            self.routes, self.stations, self.trains = self.eval_envs[self.resets % len(self.eval_envs)].get()
        elif self.mode == "render":
            self.routes, self.stations, self.trains = self.eval_envs[0].get()
        elif self.mode == "inference":
            self.routes, self.stations, self.trains = self.inference_env_bp.get()
        self.resets += 1
        for s in self.stations: s.set_input_vector(n_node_features=self.config.n_node_features, config=self.config)
        for t in self.trains: t.set_input_vector(n_node_features=self.config.n_node_features, config=self.config)
        self.trains_dict = {int(t): t for t in self.trains}
        self.stations_dict = {int(s): s for s in self.stations}
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
        edge_index_connections1 = self.routes[1]
        n_edge_connections = len(edge_index_connections0)

        edge_index_destination0 = []
        edge_index_destination1 = []

        n_passenger = 0
        edge_index_trains0 = []
        edge_index_trains1 = []
        n_trains = len(self.trains)
        for i, t in enumerate(self.trains):
            edge_index_trains0 += [i + n_stations]
            if t.destination: edge_index_trains1 += [int(t.destination)]
            else: edge_index_trains1 += [int(t.station)]
            for p in t.passengers:
                edge_index_destination0 += [i + n_stations]
                edge_index_destination1 += [int(p.destination)]
                n_passenger += 1

        for s in self.stations:
            for p in s.passengers:
                edge_index_destination0 += [int(s)]
                edge_index_destination1 += [int(p.destination)]
                n_passenger += 1

        edge_index_connections0 = np.resize(edge_index_connections0, 25_000)
        edge_index_connections1 = np.resize(edge_index_connections1, 25_000)
        edge_index_destination0 = np.asarray(edge_index_destination0)
        edge_index_destination0 = np.resize(edge_index_destination0, 25_000)
        edge_index_destination1 = np.asarray(edge_index_destination1)
        edge_index_destination1 = np.resize(edge_index_destination1, 25_000)
        edge_index_trains0 =  np.asarray(edge_index_trains0)
        edge_index_trains0 = np.resize(edge_index_trains0, 25_000)
        edge_index_trains1 = np.asarray(edge_index_trains1)
        edge_index_trains1 = np.resize(edge_index_trains1, 25_000)

        input_vectors = np.hstack([s.get_encoding() for s in self.stations+self.trains])
        input_vectors = np.resize(input_vectors, 50_000)

        return np.hstack((
            edge_index_connections0,
            edge_index_connections1,
            edge_index_destination1,  # ###weil die information von zielbahnhof zu aktuellem bahnhof fließen muss
            edge_index_destination0,
            edge_index_trains0,
            edge_index_trains1,
            input_vectors,
            n_trains,
            n_passenger,
            n_edge_connections,
            n_stations
        ))

    def get_snapshot(self):
        return dumps(self)
