import networkx
import numpy as np
import torch
import torch.nn as nn
from gym import Wrapper
from torch_geometric.nn import Linear
import math
import random
from itertools import product as cart_product
from objects import Station, PassengerGroup, Train

n_simulations = 10
n_steps = 10


class Trainer:
    def __init__(self, env, value_net, policy_net, get_ppo_action, config):
        self.env = MCTSWrapper(env)
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config
        self.mcts = MCTS(env, value_net, policy_net, self.config, get_ppo_action)
        self.train_examples_history = []

    def execute_episode(self, root):
        train_examples = []

        root = self.mcts.run(root)
        mcts_action = root.select_best_leaf().action
        next_snapshot, observation, reward, done, _ = self.env.get_result(root.snapshot, mcts_action)
        train_examples.append(([observation], root.action_probs))

        return train_examples

    def learn(self, root):
        for i in range(self.config.n_iters):
            train_examples = []
            for eps in range(self.config.n_eps):
                iteration_train_examples = self.execute_episode(root)
                train_examples.extend(iteration_train_examples)

            random.shuffle(train_examples)
            self.train(train_examples)

    def train(self, examples):
        pi_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr_pi)
        v_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.config.lr_v)

        for epoch in range(self.config.n_epochs):
            batch_idx = 0
            while batch_idx < len(examples) // self.config.batch_size:
                sample_ids = np.random.randint(len(examples), size=self.config.batch_size)
                observation, pis, vs = zip(*[examples[i] for i in sample_ids])
                target_pis = torch.FloatTensor(pis)
                target_vs = torch.FloatTensor(vs)

                pred_pis = self.policy_net.predict(observation)
                pred_vs = self.value_net.predict(observation)

                l_pi = nn.CrossEntropyLoss()(pred_pis, target_pis)
                l_pi.backward()
                pi_optim.step()

                l_v = nn.MSELoss()(pred_vs, target_vs)
                l_v.backward()
                v_optim.step()


class MCTS:
    def __init__(self, env, value_net, policy_net, config, get_ppo_action):
        self.env = MCTSWrapper(env)
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config
        self.get_ppo_action = get_ppo_action

    def predict(self, env):
        self.env = MCTSWrapper(env)
        snapshot = self.env.get_snapshot()
        root = Root(snapshot, self.env.get_observation())
        root = self.run(root)
        actions = []
        node = root
        while not node.is_leaf():
            actions.append(node.action)
            node = node.select_best_leaf()
        return actions

    def run(self, root):
        self.env.load_snapshot(root.snapshot)
        obs = root.observation
        inputs, eic, eid, eit, _ = obs  # utils.convert_observation(obs, self.config)
        actions = self.env.get_possible_mcts_actions()

        action_probs = self.policy_net(actions, inputs, eic, eid, eit)[0]
        root.expand(actions, action_probs)

        for _ in range(n_simulations):
            node = root.select_best_leaf()
            search_path = [node]
            c = 0
            done = False
            while not done:#node.is_leaf():for _ in range(n_steps):  #
                c += 1
                print(c)
                node = node.select_best_leaf()
                search_path.append(node)
                parent = node.parent
                action = node.action
                next_snapshot, obs, reward, done, info = self.env.get_result(parent.snapshot, action)
                input, eic, eid, eit, batch = obs

                self.env.load_snapshot(next_snapshot)
                actions = self.env.get_possible_mcts_actions()
                action_probs = self.policy_net(actions, input, eic, eid, eit)[0]
                node.expand(actions, action_probs)

                value = self.value_net(input, eic, eid, eit, batch)
                node.value = value
                self.backpropagate(search_path, value)


        return root

    def backpropagate(self, search_path, value):
        if isinstance(value, torch.Tensor): value = value.detach().numpy()
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1


class MCTSWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_snapshot(self):
        # edges = self.routes
        # edges = list(zip(edges[0], edges[1]))
        # g = networkx.Graph()
        # for edge in edges:
        #     g.add_edge(edge[0], edge[1])
        text = {
            "routes": self.env.routes,  # .get_all_routes(),#[route for route in self.env.routes],
            "trains": [{"station": int(train.station), "capacity": train.capacity} for train in self.env.trains],
            "passengers": [],
            "stations": [],
            "step_count": self.env.step_count}
        # "shortest_paths": self.env.shortest_path_lenghts}

        for st in self.env.trains + self.env.stations:
            for p in st.passengers:
                text["passengers"].append({"destination": int(p.destination),
                                           "n_people": p.n_people,
                                           "target_time": p.target_time,
                                           "start_station": int(p.start_station)})

        for station in self.env.stations:
            text["stations"].append({"name": station.name, "capacity": station.capacity,
                                     "reachable_stops": [int(s) for s in station.reachable_stops]})

        return text

    def load_snapshot(self, env_dict):
        # routes = Routes()

        stations_dict = {}
        for station in env_dict["stations"]:
            stations_dict[int(station["name"])] = Station(station["capacity"], station["name"])

        stations_list = [s for s in stations_dict.values()]

        for i, station in enumerate(stations_list):
            reachable_stops = env_dict["stations"][i]["reachable_stops"]
            for s in reachable_stops:
                station.reachable_stops.append(stations_list[int(s)])

        passengers = []
        for passenger in env_dict["passengers"]:
            pg = PassengerGroup(stations_dict[int(passenger["destination"])],
                                passenger["n_people"],
                                passenger["target_time"],
                                stations_dict[int(passenger["start_station"])])
            passengers.append(pg)
            station = stations_dict[int(passenger["start_station"])]
            station.passengers.append(pg)

        trains = []
        for i, train in enumerate(env_dict["trains"]):
            trains.append(Train(stations_dict[int(train["station"])], train["capacity"], name=str(i)))

        # self.env.reset()
        self.env.passengers = passengers
        self.env.step_count = env_dict["step_count"]
        self.env.routes = env_dict["routes"]  # routes.get_all_routes()
        self.env.trains = trains
        self.env.stations = stations_list
        for st in self.env.trains + self.env.stations: st.set_input_vector(self.config)

        self.env.trains_dict = {int(t): t for t in self.trains}
        self.env.stations_dict = stations_dict  # {int(s): s for s in self.stations}

        self.env.shortest_path_lenghts = self.env.init_shortest_path_lengths
        # self.env.min_steps_to_go = self.env.get_min_steps_to_go()
        self.env.active_passengers = sum([len(s.passengers) for s in self.env.stations + self.env.trains])
        self.env.step_count = 0

    def step(self, mcts_action):
        return self.env.step(mcts_action)

    def get_result(self, snapshot, mcts_action):
        self.load_snapshot(snapshot)
        observation, reward, done, info = self.step(mcts_action)
        next_snapshot = self.get_snapshot()

        return next_snapshot, observation, reward, done, info

    def get_possible_mcts_actions(self):
        # trains_start = [int(t.station) for t in self.env.trains]
        # g = self.env.graph
        actions = [[(int(t), int(d)) for d in t.station.reachable_stops] for t in self.env.trains]
        actions = list(cart_product(*actions))
        actions = torch.tensor(actions)
        # actions = actions.swapaxes(1, 0)
        return actions


class PolicyNet(nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        hidden_neurons = config.hidden_neurons
        convclass = config.conv
        self.conv1 = convclass(config.n_node_features, config.hidden_neurons, aggr=config.aggr_con)
        self.conv2 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv3 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_dest)
        self.conv4 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv5 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.lins = [Linear(hidden_neurons, hidden_neurons) for _ in range(config.n_lin_policy)]
        self.lin1 = Linear(hidden_neurons, config.action_vector_size)
        self.softmax = nn.Softmax(dim=1)
        self.config = config
        self.n = config.action_vector_size // 2
        self.dest_vecs = None
        self.start_vecs = None

    def forward(self, actions, obs, eic, eid, eit):
        actions, obs = actions.unsqueeze(0), obs.unsqueeze(0)
        start_vecs, dest_vecs = self.calc_probs(obs, eic, eid, eit)
        x = self.get_prob(actions, start_vecs, dest_vecs)

        return x

    def calc_probs(self, x, eic, eid, eit):
        x = self.conv1(x, eic)
        # x = self.activation(x)
        x = self.conv2(x, eit)
        for _ in range(self.config.it_b4_dest):
            x = self.conv3(x, eic)
            # x = self.activation(x)  #
        x = self.conv4(x, eid)
        # x = self.activation(x)

        for _ in range(self.config.it_aft_dest):
            x = self.conv5(x, eic)
            # x = self.activation(x)

        for lin in self.lins:
            x = lin(x)
        x = self.lin1(x)

        dest_vecs = x[:, :, self.n:]  # x[:, self.n:]#
        start_vecs = x[:, :, :self.n]  # x[:, :self.n]#

        return start_vecs, dest_vecs

    def get_prob(self, actions, start_vecs, dest_vecs):
        starts = start_vecs[:, actions[:, :, :, 0].flatten()]  # batches, action, stations, bool_starting
        dests = dest_vecs[:, actions[:, :, :, 1].flatten()]
        probs = torch.einsum('bij,bij->bi', starts, dests)  # Einstein Summation :)
        probs = self.softmax(probs)
        return probs


class Node:
    nodes = set()

    def __init__(self, parent, action, snapshot, prior):
        self.parent = parent
        self.action = action
        self.snapshot = snapshot
        self.prior = prior if not isinstance(prior, torch.Tensor) else prior.detach().numpy()
        self.children = set()
        self.visit_count = 0
        self.value_sum = 0
        self.action_probs = None

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_ucb_score(self):
        prior_score = self.prior * math.sqrt(self.parent.visit_count) / (self.visit_count + 1)
        return prior_score + self.value_sum

    def select_best_leaf(self):
        if self.is_leaf(): return self
        best_child = max([(c, c.get_ucb_score()) for c in self.children], key=lambda x: x[1])[0]
        return best_child.select_best_leaf()

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            node = Node(self, action, self.snapshot, prior)
            if node not in Node.nodes:
                Node.nodes.add(node)
                self.children.add(node)

    def __hash__(self):
        return hash(((p.station.name for p in self.snapshot["passengers"]),
                     (t.station.name for t in self.snapshot["trains"])))


class Root(Node):
    def __init__(self, snapshot, observation):
        super().__init__(None, None, snapshot, 1)
        self.observation = observation
