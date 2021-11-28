from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envforreal2 import AbfahrtEnv
from gym.spaces.box import Box
import numpy as np
from tqdm import tqdm
import nets.netsgat as net
import torch


def learningrate(_):
    return 10 ** -3


# def eval_model(model):
#     # print("\n === STARTING EVALUATION === \n ===============================")
#     obs = env.reset()
#     n_steps = 20
#     for step in tqdm(range(n_steps), disable=True):
#         # print(obs)
#         action = model.predict(torch.Tensor(obs), )  # , deterministic=True)
#
#         # action = action.flatten()
#         obs, reward, done, info = env.step(action)
#         # print('obs=', obs, 'reward=', reward, 'done=', done)
#         #        env.render(mode='console')
#         if done:
#             # Note that the VecEnv resets automatically
#             # when a done signal is encountered
#
#             # print("Goal reached!", "reward=", reward)
#             break
#     return step + 1


BATCH_SIZE = 20
N_STEPS = 200
TOTAL_STEPS = BATCH_SIZE * N_STEPS * 1
n_episodes = 1
N_ENVS = 1

VERBOSE = 1


if __name__ == "__main__":
    observation_space = Box(0, 1, shape=(5, 10), dtype=np.float32)
    action_space = Box(-1, +1, shape=(20,), dtype=np.float32)

    env = AbfahrtEnv(observation_space, action_space)
    env.reset()
    #check_env(env)
    multi_env = make_vec_env(lambda: env, n_envs=N_ENVS)

    model = PPO(
        net.CustomActorCriticPolicy,
        multi_env,
        vf_coef=0.05,
        verbose=VERBOSE,
        learning_rate=learningrate,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS
    )

    model.learn(TOTAL_STEPS)
    model.save("./finaltest")
