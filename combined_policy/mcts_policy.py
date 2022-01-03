import numpy as np
import torch
import torch.nn as nn
from dill import dumps, loads
from gym import Wrapper
from torch_geometric.nn import Linear
import math
import random
from itertools import product as cart_product

import utils

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

    def predict(self, snapshot, observation):
        root = Root(snapshot, observation)
        root = self.mcts.run(root)
        node = root
        action = []
        while not node.is_leaf():
            node = node.select_best_leaf()
            action.append(node.action)

        return list(reversed(action))

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

    def run(self, root):
        self.env.load_snapshot(root.snapshot)
        obs = root.observation
        inputs, eic, eid, eit, _ = utils.convert_observation(obs, self.config)
        actions = self.env.get_possible_mcts_actions()

        action_probs = self.policy_net(actions, inputs, eic, eid, eit)[0]
        root.expand(actions, action_probs)

        for _ in range(n_simulations):
            node = root
            search_path = [node]
            for _ in range(n_steps):
                node = node.select_best_leaf()
                action = node.action
                search_path.append(node)

            parent = search_path[-2]

            next_snapshot, obs, reward, done, info = self.env.get_result(parent.snapshot, action)
            # next actions
            self.env.load_snapshot(next_snapshot)
            actions = self.env.get_possible_mcts_actions()
            input, eic, eid, eit, batch = utils.convert_observation(obs, self.config)
            action_probs = self.policy_net(actions, input, eic, eid, eit)[0]
            value = self.value_net(input, eic, eid, eit, batch)
            node.value = value

            node.expand(actions, action_probs)

            self.backpropagate(search_path, value)

        return root

    @staticmethod
    def backpropagate(search_path, value):
        if isinstance(value, torch.Tensor): value = value.detach().numpy()
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1


class MCTSWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_snapshot(self):
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        self.env = loads(snapshot)

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

        dest_vecs = x[:, :, self.n:]#x[:, self.n:]#
        start_vecs = x[:, :, :self.n] #x[:, :self.n]#

        return start_vecs, dest_vecs

    def get_prob(self, actions, start_vecs, dest_vecs):
        starts = start_vecs[:, actions[:, :, :, 0].flatten()] # batches, action, stations, bool_starting
        dests = dest_vecs[:, actions[:, :, :, 1].flatten()]
        probs = torch.einsum('bij,bij->bi', starts, dests) # Einstein Summation :)
        probs = self.softmax(probs)
        return probs



class Node:
    parent = None
    qvalue_sum = 0.
    times_visited = 0

    def __init__(self, parent, action, snapshot, prior):
        self.parent = parent
        self.action = action
        self.snapshot = snapshot
        self.prior = prior if isinstance(prior, np.ndarray) else prior.detach().numpy()
        self.children = set()
        self.visit_count = 0
        self.value_sum = 0
        self.action_probs = None

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    # def get_qvalue_estimate(self):
    #     return self.q_predicted / self.times_visited if self.times_visited != 0 else 0

    def get_ucb_score(self):
        prior_score = self.prior * math.sqrt(self.parent.visit_count) / (self.visit_count + 1)
        return prior_score + self.value_sum

    def select_best_leaf(self):
        if self.is_leaf(): return self
        children = list(self.children)
        best_child = max([(c, c.get_ucb_score()) for c in self.children], key=lambda x: x[1])[0]
        return best_child.select_best_leaf()

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            node = Node(self, action, self.snapshot, prior)
            self.children.add(node)

        return self.select_best_leaf()

    # def propagate(self, child_qvalue):
    #     qvalue = self.immediate_reward + child_qvalue
    #     self.qvalue_sum += qvalue
    #     self.times_visited += 1
    #
    #     if not self.is_root(): self.parent.propagate(qvalue)


class Root(Node):
    def __init__(self, snapshot, observation):
        self.parent = self.action = None
        self.children = set()
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done = False
        self.visit_count = 0
        self.value_sum = 0
        self.action_probs = None
