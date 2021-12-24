import gym
from gym.core import Wrapper
from pickle import dumps, loads
from collections import namedtuple


def plan_mcts(root, n_iters=10):
    """
    builds tree with monte-carlo tree search for n_iters iterations
    :param root: tree node to plan from
    :param n_iters: how many select-expand-simulate-propagete loops to make
    """
    for _ in range(n_iters):
        node = Node(47, 47)
        # node = < YOUR
        # CODE: select
        # best
        # leaf >

        if node.is_done:
            # All rollouts from a terminal node are empty, and thus have 0 reward.
            node.propagate(0)
        else: pass
            # Expand the best leaf. Perform a rollout from it. Propagate the results upwards.
            # Note that here you have some leeway in choosing where to propagate from.
            # Any reasonable choice should work.

            # < YOUR
            # CODE >


class WithSnapshots(Wrapper):
    def get_snapshot(self):
        if self.unwrapped.viewer is not None:
            self.unwrapped.viewer.close()
            self.unwrapped.viewer = None
        return dumps(self.env)

    def load_snapshot(self, snapshot, render=False):
        assert not hasattr(self, "_monitor") or hasattr(
            self.env, "_monitor"), "can't backtrack while recording"

        if render:
            self.render()  # close popup windows since we can't load into them
            self.close()
        self.env = loads(snapshot)

    def get_result(self, snapshot, action):
        pass


class Node:
    parent = None  # parent Node
    qvalue_sum = 0.  # sum of Q-values from all visits (numerator)
    times_visited = 0  # counter of visits (denominator)

    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = set()  # set of child nodes

        # get action outcome and save it
        res = env.get_result(parent.snapshot, action)
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

        # compute ucb-1 additive component (to be added to mean value)
        # hint: you can use self.parent.times_visited for N times node was considered,
        # and self.times_visited for n times it was visited
        U = 47
        return self.get_qvalue_estimate() + scale * U

    def select_best_leaf(self):
        if self.is_leaf():
            return self

        children = self.children

        # Select the child node with the highest UCB score. You might want to implement some heuristics
        # to break ties in a smart way, although CartPole should work just fine without them.
        best_child =

        return best_child.select_best_leaf()

    def expand(self):
        assert not self.is_done, "can't expand from terminal state"
        for action in range(n_actions):
            self.children.add(Node(self, action))

        # If you have implemented any heuristics in select_best_leaf(), they will be used here.
        # Otherwise, this is equivalent to picking some undefined newly created child node.
        return self.select_best_leaf()

    def rollout(self, t_max=10 ** 4):
        # set env into the appropriate state
        env.load_snapshot(self.snapshot)
        obs = self.observation
        is_done = self.is_done

        # < YOUR
        # CODE: perform
        # rollout and compute
        # reward >

       # return rollout_reward

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
    def __init__(self, snapshot, observation):
        self.parent = self.action = None
        self.children = set()  # set of child nodes

        # root: load snapshot and observation
        self.snapshot = snapshot
        self.observation = observation
        self.immediate_reward = 0
        self.is_done = False

    @staticmethod
    def from_node(node):
        root = Root(node.snapshot, node.observation)
        # copy data
        copied_fields = ["qvalue_sum", "times_visited", "children", "is_done"]
        for field in copied_fields:
            setattr(root, field, getattr(node, field))
        return root