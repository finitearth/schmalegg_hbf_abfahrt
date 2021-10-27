from stable_baselines3 import PPO, A2C, DQN, SAC  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from env import AbfahrtEnv
import policy
from gym.spaces.box import Box
from gym.spaces.space import Space
import stable_baselines3
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env


def learningrate(_):
    return 10 ** -3

BATCH_SIZE = 64
N_STEPS = 64
KA_WAS_DAS = BATCH_SIZE * N_STEPS
n_episodes = 1
N_ENVS = 2
if __name__ == "__main__":
    observation_space = Box(float("-inf"), float("inf"), shape=(50,), dtype=np.float32)
    action_space = Box(-1, 1, shape=(20,), dtype=np.float32)
    # Instantiate the env
    env = AbfahrtEnv(observation_space, action_space)
    check_env(env)
    env = make_vec_env(lambda: env, n_envs=N_ENVS)

    # Train the agent
    pol = policy.CustomActorCriticPolicy(observation_space, action_space, learningrate)

    model = PPO(policy.CustomActorCriticPolicy, env, verbose=2, batch_size=BATCH_SIZE, n_steps=N_STEPS)

    # model.learn(KA_WAS_DAS)
    model.learn(total_timesteps=int(KA_WAS_DAS))

    # print(evaluate_policy(model, env))
# Test the trained agent
#     obs = env.reset()
#     n_steps = 1
#     for step in range(n_steps):
#         action, _ = model.predict(obs, deterministic=True)
#         print("Step {}".format(step + 1))
#         print("Action: ", action)
#         obs, reward, done, info = env.step(action)
#         print('obs=', obs, 'reward=', reward, 'done=', done)
# #        env.render(mode='console')
#         if done:
#             # Note that the VecEnv resets automatically
#             # when a done signal is encountered
#             print("Goal reached!", "reward=", reward)
#             break
