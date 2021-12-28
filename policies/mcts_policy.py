import random
import numpy as np
import torch
from gym.core import Wrapper
from dill import dumps, loads
from math import log, sqrt
from itertools import product as cart_product
from tqdm import tqdm
import objects


def something(observation, env, ppo_model):
    env = loads(dumps(env))
    env.using_mcts = True
    env = MCTSEnv(env, ppo_model)
    snapshot = env.get_snapshot()
    root = Root(snapshot, observation, env, ppo_model)
    plan_mcts(root)


def plan_mcts(root, n_iters=10):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in tqdm(range(n_iters)):
        node = root.select_best_leaf()
        if node.is_done:
            node.propagate(0)
        else:
            node.expand()
            reward = node.rollout()
            node.propagate(reward)


class MCTSEnv(Wrapper):
    def __init__(self, env, ppo_model):
        super().__init__(env)
        self.ppo_model = ppo_model
        self.nodes = set()

    def get_snapshot(self):
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        assert not hasattr(self, "_monitor") or hasattr(
            self.env, "_monitor"), "can't backtrack while recording"
        self.env = loads(snapshot)

    def get_result(self, snapshot, mcts_action):
        self.load_snapshot(snapshot)
        observation = np.asarray([self.env.get_observation()])
        with torch.no_grad():
            ppo_action = self.ppo_model.predict(observation)[0][0]

        observation, reward, done, info = self.step([ppo_action, mcts_action])
        next_snapshot = self.get_snapshot()

        return next_snapshot, observation, reward, done, info

    def get_possible_mcts_actions(self):
        actions = [[(int(t), int(d)) for d in t.station.reachable_stops] for t in self.trains]
        actions = cart_product(*actions)
        return actions

    def __eq__(self, other):
        self_train_station = set(int(t.station) for t in self.env.trains)
        other_train_station = set(int(t.station) for t in other.env.trains)

        self_pass_station = set()
        for st in self.trains + self.stations:
            s = st.destination if isinstance(st, objects.Train) else st
            for p in st.passengers:
                a = int(s)
                b = int(p.destination)
                self_pass_station.add((a, b))

        other_pass_station = set()
        for st in other.trains + other.stations:
            s = st.destination if isinstance(st, objects.Train) else st
            for p in st.passengers:
                a = int(s)
                b = int(p.destination)
                other_pass_station.add((a, b))

        is_equal = self_train_station == other_train_station and self_pass_station == other_pass_station

        return is_equal


class Node:
    parent = None
    qvalue_sum = 0.
    times_visited = 0

    def __init__(self, parent, action, env, snapshot, ppo_model):
        self.parent = parent
        self.action = action
        self.env = env
        self.snapshot = snapshot
        if self in self.env.nodes:
            self.is_done = True
            self.immediate_reward = -100
        else:
            self.env.nodes.add(self)

        self.ppo_model = ppo_model
        self.children = set()


    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_qvalue_estimate(self):
        return self.qvalue_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e100):
        if self.times_visited == 0:
            return max_value

        u = sqrt(2 * log(self.parent.times_visited) / self.times_visited)
        return self.get_qvalue_estimate() + scale * u

    def select_best_leaf(self):
        if self.is_leaf():
            return self
        children = self.children
        best_child = sorted(list(children), key=lambda x: x.ucb_score())[0]

        return best_child.select_best_leaf()

    def expand(self):
        assert not self.is_done, "can't expand from terminal state"
        actions = self.env.get_possible_mcts_actions()
        snapshot = self.env.get_snapshot()
        for action in actions:
            node = Node(self, action, self.env, snapshot, self.ppo_model)
            self.children.add(node)

        return self.select_best_leaf()

    def rollout(self, t_max=10):
        self.env.load_snapshot(self.snapshot)
        rollout_reward = self.immediate_reward

        if self.is_done:
            return rollout_reward
        snapshot = self.env.get_snapshot()
        for _ in range(t_max):
            mcts_action = random.choice(list(self.env.get_possible_mcts_actions()))
            snapshot, _, reward, is_done, _ = self.env.get_result(snapshot, mcts_action)
            rollout_reward += reward
            if is_done: break

        return rollout_reward

    def propagate(self, child_qvalue):
        res = self.env.get_result(self.parent.snapshot, self.action)
        self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res
        qvalue = self.immediate_reward + child_qvalue

        self.qvalue_sum += qvalue
        self.times_visited += 1

        if not self.is_root():
            self.parent.propagate(qvalue)

    def safe_delete(self):
        del self.parent
        for child in self.children:
            child.safe_delete()
            del child


class Root(Node):
    def __init__(self, snapshot, observation, env, ppo_model):
        self.env = env
        self.parent = self.action = None
        self.ppo_model = ppo_model
        self.children = set()
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done = False

    def from_node(self, node):
        root = Root(node.snapshot, node.observation, self.env, self.ppo_model)
        copied_fields = ["qvalue_sum", "times_visited", "children", "is_done"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))
        return root
