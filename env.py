import glob
import io

import gym
import networkx
import networkx as nx
import torch
import numpy as np
from PIL import ImageDraw, Image
from gym.spaces import Box
from matplotlib import pyplot as plt

import objects
import utils
from objects import EnvBlueprint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AbfahrtEnv(gym.Env):
    def __init__(self, config, mode="eval", using_mcts=False):
        super(AbfahrtEnv, self).__init__()
        self.init_shortest_path_lengths = None
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
        self.step_count += 1
        # print(self.trains[0].station.__int__(),self.step_count)
        # print(self.active_passengers)
        if self.training == "ppo":
            ppo_action = action
            mcts_action = None
        elif self.training == "mcts":
            ppo_action = None
            mcts_action = action
        else: raise ValueError("._:")
        n = self.config.action_vector_size//2
        if ppo_action is None:
            ppo_action = self.get_ppo_action(self.get_observation())

        if mcts_action is None and (self.mcts_list is None or len(self.mcts_list) < 2):
            self.training = "mcts"
            self.mcts_list = self.get_mcts_action(self)
            self.mcts_list.pop(0)
            self.training = "ppo"
        elif mcts_action is None and self.mcts_list is not None:
            mcts_action = self.mcts_list[1]
            self.mcts_list.pop(0)

        for i, station in enumerate(self.stations):
            station.vector = ppo_action[i * self.action_vector_size:(i + 1) * self.action_vector_size]

        self.rerouting_trains(mcts_action)

        for train in self.trains: # onboarding
            if not train.reached_next_stop(): continue
            for p in train.passengers: train.deboard(p)
            while len(train.passengers) < train.capacity and len(train.station.passengers) > 0:
                try:
                    dot_products = [train.station.vector[n:] @ p.destination.vector[:n] for p in train.station.passengers]
                except Exception as e:
                    print(len(ppo_action.shape))
                    raise e
                idx = np.argmax(dot_products)

                if True:#dot_products[idx]:
                    train.onboard(train.station.passengers[idx])
                else: break



        min_steps_to_go = self.get_min_steps_to_go()
        reward = self.config.reward_step_closer * (self.min_steps_to_go - min_steps_to_go)
        self.min_steps_to_go = min_steps_to_go

        active_passengers = sum([len(st.passengers) for st in self.stations+self.trains])
        reward += (self.active_passengers-active_passengers)*self.config.reward_reached_dest+self.config.reward_per_step
        self.active_passengers = active_passengers
        done = bool(active_passengers == 0)
        # if self.step_count > 500: # stop after 500 steps, because aint nobody got time for that
        #     done = True
        #     reward = -5
        if done: print(":)")
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

    def get_min_steps_to_go(self):
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
        for st in self.stations+self.trains: st.set_input_vector(config=self.config)

        self.trains_dict = {int(t): t for t in self.trains}
        self.stations_dict = {int(s): s for s in self.stations}
        edges = self.routes
        edges = list(zip(edges[0], edges[1]))
        g = networkx.Graph()
        for edge in edges:
            g.add_edge(edge[0], edge[1])

        self.init_shortest_path_lengths = dict(networkx.shortest_path_length(g))
        self.shortest_path_lenghts = self.init_shortest_path_lengths

        self.min_steps_to_go = self.get_min_steps_to_go()
        self.active_passengers = sum([len(s.passengers) for s in self.stations])
        self.step_count = 0
        return self.get_observation()

    def get_observation(self, return_type="array"):
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

        input_vectors = np.hstack([s.get_encoding() for s in self.stations + self.trains])

        if self.training != "mcts":
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

            input_vectors = np.resize(input_vectors, 50_000)
            assert None not in input_vectors, f"None in input_vectors in indices: {np.where(None == input_vectors)}"

            obs = np.hstack((
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

            assert None not in obs, f"None in obs in indices: {np.where(None == obs)}"
            return obs
        else:
            eit = np.vstack((edge_index_trains0, edge_index_trains1))
            eic = np.vstack((edge_index_connections0, edge_index_connections1))
            eid = np.vstack((edge_index_destination1, edge_index_destination0))
            edge_index_connections = torch.Tensor(eic).long()
            edge_index_trains = torch.Tensor(eit).long()
            edge_index_destinations = torch.Tensor(eid).long()
            input_vectors = torch.Tensor(input_vectors).float()
            input_vectors = torch.reshape(input_vectors, ((n_stations + n_trains), self.config.n_node_features)).float()

            batch = torch.zeros(len(input_vectors), dtype=torch.int64)

            return input_vectors.to(device), edge_index_connections.to(device), edge_index_destinations.to(device), edge_index_trains.to(device), batch.to(device)
        
    def render(self, mode="human", **kwargs):
            routes = self.routes
            trains = self.trains
            stations = self.stations
            graph = list(zip(routes[0], routes[1]))
            nx_graph = nx.Graph()
            for node in set(routes[0]):
                nx_graph.add_node(node)
            train_colors = ["red", "green", "yellow", "pink", "grey"]
            train_to_colors = {}
            for i, t in enumerate(trains):
                train_to_colors[t] = train_colors[i]
            for source, target in graph:
                nx_graph.add_edge(source, target)
            color_map = ["blue"] * len(set(routes[0]))
            for t in trains:
                station = int(t.station)
                color_map[station] = train_to_colors[t]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            nx.draw_kamada_kawai(nx_graph, with_labels=True, ax=ax, node_color=color_map)

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')

            im = Image.open(img_buf)

            black = (0, 0, 0)

            # d = ImageDraw.Draw(im)
            # d.text((28, 36), f"Iteration {self.n_calls}\nStep {step}", fill=black)

            point_map = {0: (333, 323), 1: (533, 145), 2: (355, 100), 3: (120, 364), 4: (177, 266), 5: (365, 214),
                         6: (470, 409)}
            for s in stations:
                s_ = int(s)
                # if s.vector is not None:
                #     p0, p1 = int(s.vector[0] * 50), int(s.vector[1] * 50)
                #     d, im = utils.draw_arrow(im, (point_map[s_][0] - 50, point_map[s_][1]),
                #                              (point_map[s_][0] - 50 + p0, point_map[s_][1] + p1), color=(0, 0, 255),
                #                              thickness=3)
                #     p0, p1 = int(s.vector[2] * 50), int(s.vector[3] * 50)
                #     d, im = utils.draw_arrow(im, (point_map[s_][0] - 100, point_map[s_][1]),
                #                              (point_map[s_][0] - 100 + p0, point_map[s_][1] + p1), color=(255, 0, 0),
                #                              thickness=3)
                if s.passengers:
                    d, im = utils.draw_arrow(im, point_map[s_], (point_map[s_][0], point_map[s_][1] + 25))
                    for i, p in enumerate(s.passengers, start=2):
                        d.text((point_map[s_][0] - 50, point_map[s_][1] + i * 12),
                               f"In Station -> {int(p.destination)}", fill=black)

            for t in trains:
                if t.passengers:
                    s_ = int(t.station)
                    d, im = utils.draw_arrow(im, point_map[s_], (point_map[s_][0], point_map[s_][1] + 25))
                    for i, p in enumerate(t.passengers, start=2):
                        d.text((point_map[s_][0] - 50, point_map[s_][1] + i * 12), f"In Train -> {int(p.destination)}",
                               fill=black)
            im = im.convert('RGB')
            plt.imshow(im)
            plt.show()
