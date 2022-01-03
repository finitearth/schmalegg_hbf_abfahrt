import policies.mcts_policy as mcts
import utils
from objects import EnvBlueprint
from env import AbfahrtEnv
from policies import ppo_policy
import networkx as nx


config = utils.ConfigParams(wandb_config=None)
env_bp = EnvBlueprint()
env_bp.read_txt("input/input_large.txt")
env_bp.get()
env = AbfahrtEnv(config, mode="inference")
# ppo_model = ppo_policy.get_model(env, config)
# # ppo_model.load("models/v0")
# env.inference_env_bp = env_bp
# observation = env.reset()
# mcts.something(observation, env, ppo_model)
g = env_bp.graph
# print(g)
c = 0
N = 1
# for n in list(g.edges()):
#     c+= len(list(nx.all_neighbors(g, n)))

print(len(list(g.edges())))
print(c/N)
# print(len(list(g)))/
