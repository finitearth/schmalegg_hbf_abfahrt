import policies.mcts_policy as mcts
import utils
from objects import EnvBlueprint
from env import AbfahrtEnv
from policies import ppo_policy
import networkx as nx


config = utils.ConfigParams(wandb_config=None)
env_bp = EnvBlueprint()
env_bp.random(n_max_stations=7)#read_txt("input/input_large.txt")
env_bp.get()
# env = AbfahrtEnv(config, mode="inference")
# # ppo_model = ppo_policy.get_model(env, config)
# # # ppo_model.load("models/v0")
# env.inference_env_bp = env_bp
# # observation = env.reset()
# # mcts.something(observation, env, ppo_model)
# env_bp.render()
g = (env_bp.graph)
d = nx.spring_layout(g, dim=2)#, key=lambda x: x.key()))
i = 0
try:
    while True:
        print(f"P_{i}={d[i]}")
        i+=1
except:
    pass
env_bp.render()





