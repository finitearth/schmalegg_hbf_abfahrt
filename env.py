import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
from gym.spaces import box

import generate_random
import objects


class AbfahrtEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super(AbfahrtEnv, self).__init__()
        self.passengers = []
        self.stations = []
        self.trains = []
        self.observation_space = observation_space
        self.action_space = action_space


    def step(self, action):
        reward = 0
        done = True
        info = {"huhu kann man lesen?": "Nein"}

        for passenger in self.passengers:
            if not passenger.reached_destination():
                reward -= 1
                done = False

        observation = self.get_observation()
        return observation, reward, done, info

    def reset(self):
        self.stations, self.passengers, self.trains = generate_random.generate_random_env()

        return self.get_observation()

    def render(self, mode="human"):
        raise NotImplementedError

    def get_observation(self):
        observation = []
        for station in self.stations:
            observation.append(station.getencoding())
        observation = np.asarray(observation, dtype=np.float32).flatten()
        self.observation = observation

        return observation

    def close(self):
        pass


class GoLeftEnv(gym.Env):
    """
  Custom Environment that follows gym interface.
  This is a simple env where the agent must learn to go always left.
  """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10):
        super(GoLeftEnv, self).__init__()

        # Size of the 1D-grid
        self.grid_size = grid_size
        # Initialize the agent at the right of the grid
        self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(1,), dtype=np.float32)

    def reset(self):
        """
    Important: the observation must be a numpy array
    :return: (np.array)
    """
        # Initialize the agent at the right of the grid
        self.agent_pos = self.grid_size - 1
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = bool(self.agent_pos == 0)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
