from copy import deepcopy

import torch
import torch.nn as nn
from gym import Wrapper
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from torch_geometric.nn import Linear
import math
from itertools import product as cart_product

from torch_geometric.utils import add_self_loops, to_dense_batch

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
            if node.observation is not None:
                input, eic, eit, eid, batch = node.observation
                actions = node.possible_actions
                best_action = best_child.action
                # one_hot = torch.where(actions[:, :]==best_action[None,:], True, False).all(dim=-1).flatten(start_dim=-2)*1.
                target = (actions == best_action).all(dim=-1).nonzero(as_tuple=True)[0]
                pi_example = PiData(x=input, c_edge_index=eic, t_edge_index=eit, d_edge_index=eid, actions=actions, target=target)
                pi_examples.append(pi_example)

                value = node.value_sum
                v_example = VData(x=input, c_edge_index=eic, t_edge_index=eit, d_edge_index=eid, target=value)
                v_examples.append(v_example)

                node = best_child

        return pi_examples, v_examples

    def train(self, pi_examples, v_examples):
        pi_data_loader = DataLoader(pi_examples, batch_size=32, shuffle=True, collate_fn=utils.collate) #self.config.batch_size_pi
        v_data_loader = PyGDataLoader(v_examples, batch_size=32, shuffle=True) #self.config.batch_size_v
        print(f"Batchcount: {len(v_data_loader)}")
        l_pi_function = nn.CrossEntropyLoss()
        l_v_function = nn.MSELoss()
        n_epochs = 8#self.config.n_epochs):
        for epoch in range(n_epochs):
            pi_losses, v_losses, pi_acc, v_expvar = [], [], [], []
            for x in iter(pi_data_loader):
                input, eic, eid, eit, actions, target_pis, batch = x.x, x.c_edge_index, x.d_edge_index, x.t_edge_index, x.actions, x.target, x.batch
                pred_pis = self.policy_net(actions, input, eic, eit, eid, batch)
                l_pi = l_pi_function(pred_pis, target_pis)
                pi_losses.append(l_pi)
                l_pi.backward()
                pred_onehot = torch.argmax(pred_pis, -1)
                pi_acc.append(sum(pred_onehot==target_pis)/len(pred_onehot))
                self.pi_optim.step()

            for x in iter(v_data_loader):
                input, eic,  eid, eit, target_vs, batch = x.x, x.c_edge_index, x.d_edge_index, x.t_edge_index, x.target, x.batch
                pred_vs = self.value_net(input, eic, eit, eid, batch)
                pred_vs = pred_vs.squeeze(1)
                l_v = l_v_function(pred_vs, target_vs.float())
                v_losses.append(l_v)
                l_v.backward()
                self.v_optim.step()
                exp_var = 1 - torch.var(pred_vs-target_vs)/torch.var(target_vs)
                v_expvar.append(exp_var)

            print(f"Epoch {epoch+1}/{n_epochs}, "
                  f"v_loss: {sum(v_losses)/len(v_losses):.2f},"
                  f" pi_loss: {sum(pi_losses)/len(pi_losses):.2f},"
                  f" pi_acc: {sum(pi_acc)/len(pi_acc):.2f}, "
                  f"v_expvar: {sum(v_expvar)/len(v_expvar):.2f}")






class PiData(Data):
    def __init__(self, x=None, c_edge_index=None, d_edge_index=None, t_edge_index=None, actions=None, target=None, **kwargs):
        super().__init__(x=x, target=target, actions=actions,
                         c_edge_index=c_edge_index, d_edge_index=d_edge_index, t_edge_index=t_edge_index, **kwargs)
        if c_edge_index is not None:
            c_edge_index, _ = add_self_loops(c_edge_index)
        # print("durchgluffe")
        # if c_edge_index is not None:
        #     self.actions = self.actions.long()# if actions else None
        #     self.target = self.target.float()#.long() #if target  else None


class VData(Data):
    def __init__(self, x=None, c_edge_index=None, d_edge_index=None, t_edge_index=None, target=None, **kwargs):
        super().__init__(x=x, target=target, **kwargs)
        if c_edge_index is not None:
            c_edge_index, _ = add_self_loops(c_edge_index)
        # print("durchgluffe")
        if c_edge_index is not None:
            self.c_edge_index = c_edge_index.long()  # if c_edge_index else None
            self.d_edge_index = d_edge_index.long()  # if d_edge_index else None
            self.t_edge_index = t_edge_index.long()  # if t_edge_index else None
            self.target = target



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
        root.possible_actions = actions
        batch = torch.zeros(inputs.shape[0], dtype=torch.long)
        with torch.no_grad():
            action_probs = self.policy_net(actions, inputs, eic, eid, eit, batch)
        root.expand(actions, action_probs)

        for _ in range(n_simulations):
            node = root.select_best_leaf()
            done = False
            while not done:  # for actions in actionss:
                node = node.select_best_leaf()
                next_snapshot, obs, reward, done, info = self.env.get_result(node)
                node.set_snapshot(next_snapshot)
                input, eic, eid, eit, batch = obs
                # print(obs)
                node.observation = obs
                with torch.no_grad():
                    value_pred = self.value_net(input, eic, eid, eit, batch)
                self.value = value_pred
                node.backpropagate(reward)
                actions = self.env.get_possible_mcts_actions(node.snapshot)
                node.possible_actions = actions
                action_probs = self.policy_net(actions, input, eic, eid, eit, batch)[0]
                node.expand(actions, action_probs)
                while True:
                    if not node.is_dead_end(): break
                    node.value_sum -= 1.
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
                        "vector": train.input_vector,
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
                                     "reachable_stops": [int(s) for s in station.reachable_stops],
                                    "vector": station.input_vector})

        return text

    def load_snapshot(self, env_dict):
        stations_dict = {}
        for station in env_dict["stations"]:
            s = Station(station["capacity"], station["name"])
            stations_dict[int(station["name"])] = s
            s.input_vector = station["vector"]
        stations_list = [s for s in stations_dict.values()]

        for i, station in stations_dict.items():
            rs = env_dict["stations"][i]["reachable_stops"]
            for s in rs:
                station.reachable_stops.append(stations_dict[int(s)])

        trains = []
        for train in env_dict["trains"]:
            t = Train(station=stations_dict[int(train["station"])], capacity=train["capacity"],
                      name=train["name"], speed=train["speed"])
            t.input_vector = train["vector"]
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
        # for st in self.env.trains + self.env.stations: st.set_input_vector(self.config)

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
        actions = [[(int(t.station), int(d)) for d in t.station.reachable_stops]
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

    def forward(self, actions, obs, eic, eid, eit, batch):
        start_vecs, dest_vecs = self.calc_probs(obs, eic, eid, eit, batch)
        x = self.get_prob(actions, start_vecs, dest_vecs)

        return x

    def calc_probs(self, x, eic, eid, eit, batch):
        x = self.conv1(x, eit)
        x = self.conv2(x, eic)
        x = self.conv3(x, eid)

        for lin in self.lins:
            x = lin(x)
        x = self.lin1(x)
        x, _ = to_dense_batch(x, batch)
        start_vecs = x[:, :, :self.n]
        dest_vecs = x[:, :, self.n:]

        return start_vecs, dest_vecs

    def get_prob(self, actions, start_vecs, dest_vecs):
        # batches, action, stations, bool_starting
        try:
            starts = start_vecs[:,  actions[:, :, 0].flatten()]  # ALARM SIEHE GET POSSIBLE ACTIONS -.-
            dests = dest_vecs[ :, actions[:, :, 1].flatten()]
        except Exception as e:
            print("-.-")
            raise e
        # Einstein Summation :)
        probs = torch.einsum('bij,bij->bj', starts, dests).to(device)

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

    def backpropagate(self, reward):
        # if isinstance(value, torch.Tensor): value = value.cpu().detach().numpy()
        self.value_sum += float(reward)
        self.visit_count += 1
        self.parent.backpropagate(reward)

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
        raise DeadEndException()


class DeadEndException(Exception):
    def __init__(self):
        super().__init__("bibabo")
