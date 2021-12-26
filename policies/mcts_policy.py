import random
import numpy as np
import torch
from gym.core import Wrapper
from dill import dumps, loads
from collections import namedtuple
from math import log, sqrt
from itertools import product as cart_product

ActionResult = namedtuple(
    "action_result", ("snapshot", "observation", "reward", "is_done", "info"))
n_actions = 10


def something(observation, env, ppo_model):
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
    for _ in range(n_iters):
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

    def get_snapshot(self):
        return dumps(self.env)

    def load_snapshot(self, snapshot):
        assert not hasattr(self, "_monitor") or hasattr(
            self.env, "_monitor"), "can't backtrack while recording"
        self.env = loads(snapshot)

    def get_result(self, snapshot, mcts_action):
        self.load_snapshot(snapshot)
        observation = np.asarray([self.get_observation()])
        with torch.no_grad():
            ppo_action = self.ppo_model.predict(observation)[0][0]
        observation, reward, done, info = self.step([ppo_action, mcts_action])
        next_snapshot = self.get_snapshot()

        return next_snapshot, observation, reward, done, info

    def get_possible_mcts_actions(self):
        actions = [[(t, d) for d in t.station.reachable_stops] for t in self.trains]
        actions = cart_product(*actions)
        return actions


class Node:
    parent = None  # parent Node
    qvalue_sum = 0.  # sum of Q-values from all visits (numerator)
    times_visited = 0  # counter of visits (denominator)

    def __init__(self, parent, action, env, ppo_model):
        self.parent = parent
        self.action = action
        self.env = env
        self.ppo_model = ppo_model
        self.children = set()  # set of child nodes
        # get action outcome and save it
        res = self.env.get_result(parent.snapshot, action)
        self.snapshot, self.observation, self.immediate_reward, self.is_done, _ = res

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def get_qvalue_estimate(self):
        return self.qvalue_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e100):
        if self.times_visited == 0:
            return max_value

        # compute ucb-1 additive component
        u = sqrt(2 * log(self.parent.times_visited) / self.times_visited)
        return self.get_qvalue_estimate() + scale * u

    def select_best_leaf(self):
        if self.is_leaf():
            return self

        children = self.children

        # Select the child node with the highest UCB score. You might want to implement some heuristics
        # to break ties in a smart way, although CartPole should work just fine without them.
        best_child = sorted(list(children), key=lambda x: x.ucb_score())[0]

        return best_child.select_best_leaf()

    def expand(self):
        assert not self.is_done, "can't expand from terminal state"
        actions = self.env.get_possible_mcts_actions()
        for action in actions:
            self.children.add(Node(self, action, self.env, self.ppo_model))

        # If you have implemented any heuristics in select_best_leaf(), they will be used here.
        # Otherwise, this is equivalent to picking some undefined newly created child node.
        return self.select_best_leaf()

    def rollout(self, t_max=10 ** 4):
        # set env into the appropriate state
        self.env.load_snapshot(self.snapshot)
        observation = self.observation
        rollout_reward = self.immediate_reward
        if self.is_done:
            return rollout_reward

        for _ in range(t_max):
            mcts_action = random.choice(self.env.get_possible_mcts_actions())
            with torch.no_grad(): ppo_action = self.ppo_model.predict([observation])
            observation, reward, is_done, _ = self.env.step([ppo_action, mcts_action])
            rollout_reward += reward
            if is_done: break

        return rollout_reward

    def propagate(self, child_qvalue):
        # compute node Q-value
        my_qvalue = self.immediate_reward + child_qvalue

        # update qvalue_sum and times_visited
        self.qvalue_sum += my_qvalue
        self.times_visited += 1

        # propagate upwards
        if not self.is_root():
            self.parent.propagate(my_qvalue)

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
        self.children = set()  # set of child nodes

        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done = False

    def from_node(self, node):
        root = Root(node.snapshot, node.observation, self.env, self.ppo_model)
        # copy data
        copied_fields = ["qvalue_sum", "times_visited", "children", "is_done"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))
        return root
