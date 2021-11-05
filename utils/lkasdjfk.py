from stable_baselines3 import PPO, A2C, DQN, SAC  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from env import AbfahrtEnv
import policy
from gym.spaces.box import Box
import numpy as np
from stable_baselines3.common.env_checker import check_env
import torch as th


print(th.tensor(1) == 1)