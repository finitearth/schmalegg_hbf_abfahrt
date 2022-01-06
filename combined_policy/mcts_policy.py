from copy import deepcopy

import networkx
from networkx.algorithms.assortativity.pairs import node_attribute_xy
import numpy as np
import torch
import torch.nn as nn
from gym import Wrapper
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Linear
import math
import random
from itertools import product as cart_product

import env
import utils
from objects import Station, PassengerGroup, Train

n_simulations = 10
n_steps = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, env, value_net, policy_net, get_ppo_action, config):
        self.env = MCTSWrapper(env)
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config
        self.mcts = MCTS(env, value_net, policy_net,
                         self.config, get_ppo_action)
        self.pi_examples = []
        self.v_examples = []
        self.pi_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr_pi)
        self.v_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.config.lr_v)

    def execute_episode(self, root):
        root = self.mcts.search(root)
        pi_examples = []
        v_examples = []
        node = root
        while not node.is_leaf():
            best_child = node.get_best_child()
            obs = node.observation
            input, eic, eit, eid = utils.convert_observation(obs, self.config)
            actions = node.actions
            best_action = best_child.action
            one_hot = (actions == best_action)*1
            pi_example = utils.CustomData(x=input, eic=eic, eit=eit, eid=eid, actions=actions, one_hot=one_hot)
            pi_examples.append(pi_example)

            value = node.value_sum
            v_example = utils.CustomData(x=input, eic=eic, eit=eit, eid=eid, value=value)
            v_examples.append(v_example)

            node = best_child

        return pi_examples, v_examples

    def train(self, pi_examples, v_examples):
        pi_data_loader = DataLoader(pi_examples, batch_size=self.config.batch_size_pi, shuffle=True)
        v_data_loader = DataLoader(v_examples, batch_size=self.config.batch_size_v, shuffle=True)

        for epoch in range(self.config.n_epochs):
            pi_losses, v_losses = [], []
            for input, eic, eit, eid, actions, target_pis in iter(pi_data_loader):
                pred_pis = self.policy_net.predict(input, eic, eit, eid, actions)
                l_pi = nn.CrossEntropyLoss()(pred_pis, target_pis)
                pi_losses.append(l_pi)
                l_pi.backward()
                self.pi_optim.step()

            for input, eic, eit, eid, target_vs in iter(v_data_loader):
                pred_vs = self.value_net.predict(input, eic, eit, eid,)
                l_v = nn.MSELoss()(pred_vs, target_vs)
                v_losses.append(l_v)
                l_v.backward()
                self.v_optim.step()

            print(f"Epoch {epoch}/{self.config.n_epochs}, v_loss: {np.mean(v_losses)}, pi_loss: {np.mean(pi_losses)}")



class MCTS:
    def __init__(self, env, value_net, policy_net, config, get_ppo_action):
        self.env = MCTSWrapper(env)
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config
        self.get_ppo_action = get_ppo_action

    def predict(self, env):
        raise NotImplementedError

    #     self.env = MCTSWrapper(env)
    #     snapshot = self.env.get_snapshot()
    #     root = Root(snapshot, self.env.get_observation())
    #     root = self.run(root)
    #     actions = []
    #     node = root
    #     while not node.is_leaf():
    #         actions.append(node.action)
    #         node = node.select_best_leaf()
    #     return actions

    def search(self, root):
        self.env.load_snapshot(root.snapshot)
        obs = root.observation
        inputs, eic, eid, eit, _ = obs  # utils.convert_observation(obs, self.config)
        actions = self.env.get_possible_mcts_actions(root.snapshot)
        action_probs = self.policy_net(actions, inputs, eic, eid, eit)[0]
        root.expand(actions, action_probs)

        for _ in range(n_simulations):
            node = root.select_best_leaf()
            done = False
            while not done:  # for actions in actionss:
                node = node.select_best_leaf()
                next_snapshot, obs, reward, done, info = self.env.get_result(node)
                node.set_snapshot(next_snapshot)
                input, eic, eid, eit, batch = obs
                node.observation = obs
                value = self.value_net(input, eic, eid, eit, batch)
                node.backpropagate(value)
                actions = self.env.get_possible_mcts_actions(node.snapshot)
                action_probs = self.policy_net(actions, input, eic, eid, eit)[0]
                node.expand(actions, action_probs)
                while True:
                    if not node.is_dead_end(): break
                    node.value_sum -= 1
                    node = node.parent
                    # continue

                # print(f"step: {node.snapshot['step_count']}; action: {node.action}")
                # self.env.render()
        Node.nodes = []
        return root


class MCTSWrapper(Wrapper):
    def __init__(self, env_):
        assert isinstance(env_, env.AbfahrtEnv), "bist du dumm"
        super().__init__(env_)

    def get_snapshot(self):
        step_count = deepcopy(self.env.step_count)
        text = {
            "routes": self.env.routes,
            "trains": [{"station": int(train.station), "capacity": train.capacity, "speed": train.speed,
                        "destination": int(train.destination) if train.destination is not None else -1,
                        "name": train.name} for train in self.env.trains],
            "passengers": [],
            "stations": [],
            "step_count": step_count}

        for st in self.env.trains + self.env.stations:
            for p in st.passengers:
                text["passengers"].append({"destination": int(p.destination),
                                           "n_people": p.n_people,
                                           "target_time": p.target_time,
                                           "current": int(st),
                                           "st": "t" if isinstance(st, Train) else "s"})

        for station in self.env.stations:
            text["stations"].append({"name": station.name, "capacity": station.capacity,
                                     "reachable_stops": [int(s) for s in station.reachable_stops]})
            # "vector": station.input_vector})

        return text

    def load_snapshot(self, env_dict):
        stations_dict = {}
        for station in env_dict["stations"]:
            s = Station(station["capacity"], station["name"])
            stations_dict[int(station["name"])] = s
            s.set_input_vector(self.config)
        stations_list = [s for s in stations_dict.values()]

        for i, station in stations_dict.items():
            rs = env_dict["stations"][i]["reachable_stops"]
            for s in rs:
                station.reachable_stops.append(stations_dict[int(s)])

        trains = []
        for train in env_dict["trains"]:
            t = Train(station=stations_dict[int(train["station"])], capacity=train["capacity"],
                      name=train["name"], speed=train["speed"])
            t.set_input_vector(self.config)
            destination = train["destination"]
            if destination != -1:
                t.destination = (stations_dict[int(destination)])
            trains.append(t)

        passengers = []
        for passenger in env_dict["passengers"]:
            pg = PassengerGroup(stations_dict[int(passenger["destination"])],
                                passenger["n_people"],
                                passenger["target_time"], 0)

            st = passenger["st"]
            if st == "s":
                station = stations_dict[int(passenger["current"])]
                station.passengers.append(pg)
            else:
                train = trains[int(passenger["current"])]
                train.passengers.append(pg)

            # stations_dict[int(passenger["start_station"])])

        self.env.active_passengers = sum([len(st.passengers) for st in stations_list])
        self.env.active_passengers += sum([len(st.passengers) for st in trains])
        self.env.passengers = passengers
        self.env.step_count = env_dict["step_count"]
        self.env.routes = env_dict["routes"]
        self.env.trains = trains
        self.env.stations = stations_list
        for st in self.env.trains + self.env.stations: st.set_input_vector(self.config)

        self.env.trains_dict = {int(t): t for t in self.trains}
        self.env.stations_dict = {int(s): s for s in self.stations}

        self.env.shortest_path_lenghts = self.env.init_shortest_path_lengths
        # self.env.min_steps_to_go = self.env.get_min_steps_to_go()

    def step(self, mcts_action):
        return self.env.step(mcts_action)

    def get_result(self, node):
        self.load_snapshot(node.parent.snapshot)
        observation, reward, done, info = self.step(node.action)
        next_snapshot = self.get_snapshot()
        return next_snapshot, observation, reward, done, info

    def get_possible_mcts_actions(self, snapshot):
        self.load_snapshot(snapshot)
        actions = [[(int(t), int(d)) for d in t.station.reachable_stops]
                   for t in self.env.trains]  # ALARM IM POLICYNETWORK MUSS ANDERES ALS T SEIN -.-
        actions = list(cart_product(*actions))
        actions = torch.tensor(actions)
        return actions


class PolicyNet(nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        hidden_neurons = config.hidden_neurons
        convclass = config.conv
        self.conv1 = convclass(config.n_node_features,
                               config.hidden_neurons, aggr=config.aggr_con)
        self.conv2 = convclass(config.hidden_neurons,
                               config.hidden_neurons, aggr=config.aggr_con)
        self.conv3 = convclass(config.hidden_neurons,
                               config.hidden_neurons, aggr=config.aggr_con)
        self.lins = [Linear(hidden_neurons, hidden_neurons).to(device)
                     for _ in range(2)]
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
        x = self.conv1(x, eit)
        x = self.conv2(x, eic)
        x = self.conv3(x, eid)

        for lin in self.lins:
            x = lin(x)
        x = self.lin1(x)

        dest_vecs = x[:, :, self.n:]
        start_vecs = x[:, :, :self.n]

        return start_vecs, dest_vecs

    def get_prob(self, actions, start_vecs, dest_vecs):
        # batches, action, stations, bool_starting
        starts = start_vecs[:, actions[:, :, :, 0].flatten()]  # ALARM SIEHE GET POSSIBLE ACTIONS -.-
        dests = dest_vecs[:, actions[:, :, :, 1].flatten()]
        # Einstein Summation :)
        probs = torch.einsum('bij,bij->bi', starts, dests).to(device)

        probs = self.softmax(probs)
        return probs


class Node:
    nodes = list()

    def __init__(self, parent, action, prior):
        self.observation = None
        self.expanded = False
        self.parent = parent
        self.action = action
        self.prior = prior if not isinstance(prior, torch.Tensor) else prior.cpu().detach().numpy()
        self.children = set()
        self.visit_count = 0
        self.value_sum = 0
        self.action_probs = None
        self.snapshot = None
        self.hasher = str(parent.snapshot["stations"]) + \
                      str(parent.snapshot["trains"]) + \
                      str(parent.snapshot["passengers"]) \
                      + str(self.action)  if parent else "root"

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def is_dead_end(self):
        if not self.expanded: return False
        if len(self.children) < 1: return True
        for child in self.children:
            if not child.is_dead_end(): return False
        return True

    def get_ucb_score(self):
        return self.value_sum + self.prior * math.sqrt(self.parent.visit_count / (self.visit_count + 1))

    def select_best_leaf(self):
        if self.is_leaf(): return self
        best_child = max([(c, c.get_ucb_score()) for c in self.children], key=lambda x: x[1])[0]

        return best_child.select_best_leaf()

    def get_best_child(self):
        best_child = max([(c, c.value_sum) for c in self.children], key=lambda x: x[1])[0]
        return best_child

    def expand(self, actions, priors):
        self.expanded = True
        for action, prior in zip(actions, priors):
            node = Node(self, action, prior)
            if node not in Node.nodes:
                Node.nodes.append(node)
                self.children.add(node)

    def backpropagate(self, value):
        if isinstance(value, torch.Tensor): value = value.cpu().detach().numpy()
        self.value_sum += value
        self.visit_count += 1
        self.parent.backpropagate(value)

    def __hash__(self):
        return hash(self.hasher)

    def __eq__(self, other):
        return self.hasher == other.hasher


class Root(Node):
    def __init__(self, snapshot, observation):
        super().__init__(None, None, 1)
        self.observation = observation
        self.snapshot = snapshot
        self.hasher = tuple()

    def backpropagate(self, value):
        self.value_sum += value

    def is_dead_end(self):
        # return False
        for child in self.children:
            if not child.is_dead_end(): return False
        raise ValueError("No non-dead-ends found")
