import time
from abc import ABC

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

from envforreal2 import AbfahrtEnv
from gym.spaces.box import Box
from gym.spaces.space import Space
import numpy as np
import nets.netsgat as net
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
import torch
from stable_baselines3.common.env_checker import check_env


class WandBCallback(BaseCallback):
    def __init__(self, check_freq, eval_freq=1, eval_env=None):
        super(WandBCallback, self).__init__(1)
        self.check_freq = check_freq
        # self.best_mean_reward = -np.inf
        # self.eval_freq = 2048
        self.eval_env = eval_env
        self.n_eval_episodes = 16
        self.eval_freq = N_STEPS

    def init_callback(self, model: "base_class.BaseAlgorithm") -> None:
        self.model = model

    def _on_step(self) -> bool:
        hundred_percent = False
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, episode_lengths, hundred_percent, dings = self._evaluate_policy()
            if USE_WANDB:
                wandb.log({"episode_lenghts": episode_lengths})
                wandb.log({"succesfull_episodes %": dings})
            else:
                print(f"{self.n_calls}: {episode_lengths}")

        return not hundred_percent

    def _evaluate_policy(self):
        # TODO parallel enviroments
        #  TODO batches
        # self.eval_env = DummyVecEnv([lambda: self.eval_env])

        episode_rewards = []
        episode_lengths = []
        t0 = time.perf_counter()
        for _ in range(self.n_eval_episodes):
            observation = self.eval_env.reset(mode="eval")
            current_reward = 0
            steps_taken = 0
            for i in range(100):
                observation = torch.tensor([observation]).float()

                action, _, _ = self.model.policy.forward(observation, deterministic=True)

                action = action.detach().numpy()[0]
                observation, reward, done, info = self.eval_env.step(action)
                current_reward += reward

                if done: break
                steps_taken += 1

            episode_rewards.append(current_reward)
            episode_lengths.append(steps_taken)
        time_taken = time.perf_counter() - t0
        print(f"======== EVALUATION ========= it took: {time_taken}")
        dings = np.count_nonzero(np.asarray(episode_lengths) < 100)/self.n_eval_episodes*100
        print(f"reached goal: {dings} %")
    #    if dings == 100: print(action)
        # print(episode_lengths)
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        return mean_reward, mean_length, False, dings#(np.count_nonzero(np.asarray(episode_lengths) < 200)) == 16, dings


def train():
    with wandb.init() as run:
        config = wandb.config
        LEARNING_RATE = config.learning_rate
        CLIP_RANGE = config.clip_range

        observation_space = Box(-100, +100, shape=(3203,), dtype=np.float32)
        action_space = Box(-100, +100, shape=(400,), dtype=np.float32)

        env = AbfahrtEnv(observation_space, action_space)
        env.reset()
        # check_env(env)
        multi_env = make_vec_env(lambda: env, n_envs=N_ENVS)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPO(
            net.CustomActorCriticPolicy,
            multi_env,
            vf_coef=VF_COEF,
            verbose=VERBOSE,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            n_steps=N_STEPS,
            clip_range=CLIP_RANGE,
            device=device,
            # gamma=GAMMA
            # _init_setup_model=False
        )
        eval_callback = WandBCallback(1, 1, env)

        model.learn(TOTAL_STEPS, callback=eval_callback)


VERBOSE = 1
USE_WANDB = 1

CLIP_RANGE = 0.2
VF_COEF = 0.5
BATCH_SIZE = 16
N_STEPS = 256
TOTAL_STEPS = BATCH_SIZE * N_STEPS * 1_000
n_episodes = 1
N_ENVS = 16
LEARNING_RATE = 10 ** -4
GAMMA = 0.99

if __name__ == "__main__":
    if USE_WANDB:
        #wandb.init(project="schmalegger-hbf", entity="schmalegg")
        sweep_id = "schmalegg/schmalegger-hbf/cz7uo6vd"
        wandb.agent(sweep_id, function=train, count=10)
    else:
        train()
