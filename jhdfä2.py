import policies.mcts_policy as mcts
import utils
from objects import EnvBlueprint
from env import AbfahrtEnv
from policies import ppo_policy

config = utils.ConfigParams()
env_bp = EnvBlueprint()
env_bp.read_txt("input/input5.txt")
env = AbfahrtEnv(config, mode="inference")
ppo_model = ppo_policy.get_model(env, config)
ppo_model.load("models/testing")
env.inference_env_bp = env_bp
observation = env.reset()
mcts.something(observation, env, ppo_model)
