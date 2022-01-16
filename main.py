from cmath import log
import read_input_files
import sys
import mcts.mcts
from utils import ConfigParams, StepLogger
from mcts.value_net import ValueNet
from mcts.policy_net import PolicyNet
from mcts.value_net import ValueNet
import env_tensor_for_real
import torch 

file = str(sys.argv[1])
obs = read_input_files.read_txt(file)
device = torch.device("cuda")
config = ConfigParams()
value_net = ValueNet(config=config).to(device)
policy_net = PolicyNet(config=config).to(device)

mcts=mcts.mcts.MCTS(value_net, policy_net, config)
best_action_history, best_reward_history = mcts.search(obs)
logger = StepLogger()
obs = env_tensor_for_real.init_step(obs, logger, best_action_history)
done = False
while not done:
    obs, done = env_tensor_for_real.step(obs, logger, best_action_history)
logger.save_log
