from stable_baselines3 import PPO, A2C, DQN  # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from env import AbfahrtEnv, GoLeftEnv
import policy
from gym.spaces.box import Box
from gym.spaces.space import Space
import stable_baselines3
import numpy as np

def learningrate(_):
    return 10**-3


if __name__ == "__main__":
    observation_space = Box(float("-inf"), float("inf"), shape=(50,), dtype=np.float32)
    action_space = Box(-1, 1, shape=(20,), dtype=np.float32)
    # Instantiate the env
    env = AbfahrtEnv(observation_space, action_space)

    #stable_baselines3.env_checker.check_env(env, warn=True, skip_render_check=True)
    # wrap it
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    env = make_vec_env(lambda: env, n_envs=1)

    # Train the agent

    pol = policy.CustomActorCriticPolicy(observation_space, action_space, learningrate)
    BATCH_SIZE = 3
    KA_WAS_DAS = 2
    N_STEPS = BATCH_SIZE * KA_WAS_DAS
    model = PPO(policy.CustomActorCriticPolicy, env, verbose=1, batch_size=BATCH_SIZE, n_steps=N_STEPS).learn(KA_WAS_DAS)

    # Test the trained agent
#     obs = env.reset()
#     n_steps = 1
#     for step in range(n_steps):
#         action, _ = model.predict(obs)#, deterministic=True)
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
