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
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt


def learningrate(_):
    return 10 ** -4


def eval_model():
    # print("\n === STARTING EVALUATION === \n ===============================")
    obs = env.reset()
    n_steps = 20
    for step in tqdm(range(n_steps), disable=True):
        # print(obs)
        action = model.predict(obs, deterministic=True)

        # print("\n ===  Step {}  ===".format(step + 1))
        # print("Action: ", action)
        #  print(step)
        obs, reward, done, info = env.step(action.detach().numpy())
        # print('obs=', obs, 'reward=', reward, 'done=', done)
        #        env.render(mode='console')
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break
    return step + 1


BATCH_SIZE = 10
N_STEPS = 10
KA_WAS_DAS = BATCH_SIZE * N_STEPS
n_episodes = 1
N_ENVS = 10
if __name__ == "__main__":
    observation_space = Box(0, 1, shape=(50,), dtype=np.float32)
    action_space = Box(-1, 1, shape=(20,), dtype=np.float32)
    # Instantiate the env
    env = AbfahrtEnv(observation_space, action_space)
    env.reset()
    check_env(env)
    multi_env = make_vec_env(lambda: env, n_envs=N_ENVS)

    # Train the agent
    pol = policy.CustomActorCriticPolicy  # (observation_space, action_space, learningrate)

    model = PPO(pol, multi_env, verbose=0, batch_size=BATCH_SIZE, n_steps=N_STEPS)
    steps_it_took = []
    episode = []
    for n in tqdm(range(200)):
        # model.learn(KA_WAS_DAS)
        model = model.learn(total_timesteps=int(KA_WAS_DAS))
        steps_it_took.append(eval_model())
        episode.append(n)

    plt.plot(episode, steps_it_took)
    plt.show()
