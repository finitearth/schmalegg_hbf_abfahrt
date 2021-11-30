import time

from stable_baselines3.common.env_util import make_vec_env

from enviroments.env import AbfahrtEnv
from gym.spaces.box import Box
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import torch

from policies import ppo_policy, ppo_policy2


class WandBCallback(BaseCallback):
    def __init__(self, check_freq, eval_env=None):
        super(WandBCallback, self).__init__(1)
        self.check_freq = check_freq

        self.eval_env = eval_env
        self.n_eval_episodes = 1
        self.eval_freq = N_STEPS

    def init_callback(self, model):
        self.model = model

    def _on_step(self) -> bool:
        hundred_percent = False
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []
            episode_rewards, episode_lengths, hundred_percent, dings = self._evaluate_policy()
            print(f"reached goal: {dings} %")
            print(f"{self.n_calls}: {episode_lengths}")
            if USE_WANDB:
                wandb.log({"episode_lenghts": episode_lengths})
                wandb.log({"succesfull_episodes %": dings})

        return not hundred_percent

    def _evaluate_policy(self):
        t0 = time.perf_counter()
        n_envs = 16
        episode_rewards = []
        episode_lengths = []

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(self.n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        observations = self.eval_env.reset()
        episode_starts = np.ones((self.eval_env.num_envs,), dtype=bool)
        for _ in range(100):
            observations = torch.tensor(observations).to(device)
            actions, _, _ = self.model.policy.forward(observations, deterministic=True)
            actions = actions.detach().numpy()
            observations, rewards, dones, infos = self.eval_env.step(actions)
            current_rewards += rewards
            current_lengths += 1
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    reward = rewards[i]
                    done = dones[i]
                    episode_starts[i] = done

                    if dones[i]:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0

        for done in dones:
            if not done: episode_lengths.append(100)
        hundred_percent = max(current_lengths) != 100
        time_taken = time.perf_counter() - t0
        print(f"======== EVALUATION ========= it took: {time_taken}")
        dings = np.count_nonzero(np.asarray(episode_lengths) < 100) / self.n_eval_episodes * 100
        return episode_rewards, np.mean(episode_lengths), hundred_percent, dings


def train():
    if USE_WANDB:
        with wandb.init() as run:
            config = wandb.config
            if config.policy == "ppo_policy":
                policy = ppo_policy
            elif config.policy == "ppo_policy2":
                policy = ppo_policy2
            else:
                raise NotImplementedError

            VF_COEF = config.vf_coef
            LEARNING_RATE = config.learning_rate
            GAMMA = config.gamma
            CLIP_RANGE = config.clip_range
            N_NODE_FEATURES = config.n_node_features
            ACTION_VECTOR_SIZE = config.action_vector_size
            HIDDEN_NEURONS = config.hidden_neurons
            ITERATIONS_BEFORE_DESTINATION = config.it_b4_dest
            ITERATIONS_AFTER_DESTINATION = config.it_aft_dest
            USE_BN = config.use_bn

            env = AbfahrtEnv(observation_space, action_space, action_vector_size=ACTION_VECTOR_SIZE, n_node_features=N_NODE_FEATURES)
            env.reset()

            multi_env = make_vec_env(lambda: env, n_envs=N_ENVS)
            eval_envs = make_vec_env(lambda: env, n_envs=16)

            model = policy.get_model(multi_env, vf_coef=VF_COEF, verbose=VERBOSE, learning_rate=LEARNING_RATE,
                                     batch_size=BATCH_SIZE, n_steps=N_STEPS, clip_range=CLIP_RANGE, gamma=GAMMA,
                                     n_node_features=N_NODE_FEATURES, hidden_neurons=HIDDEN_NEURONS,
                                     output_features=ACTION_VECTOR_SIZE, it_b4_dest=ITERATIONS_BEFORE_DESTINATION,
                                     it_aft_dest=ITERATIONS_AFTER_DESTINATION, use_bn=USE_BN)

            eval_callback = WandBCallback(1, 1, eval_envs)
            model.learn(TOTAL_STEPS, callback=eval_callback)

    else:
        CLIP_RANGE = 0.2
        VF_COEF = 0.5
        LEARNING_RATE = 10 ** -4
        GAMMA = 0.99

        env = AbfahrtEnv(observation_space, action_space)
        env.reset()
        # check_env(env)
        multi_env = make_vec_env(lambda: env, n_envs=N_ENVS)

        model = ppo_policy.get_model(multi_env, vf_coef=VF_COEF, verbose=VERBOSE, learning_rate=LEARNING_RATE,
                                     batch_size=BATCH_SIZE, n_steps=N_STEPS, clip_range=CLIP_RANGE, gamma=GAMMA)
        eval_envs = make_vec_env(lambda: env, n_envs=16)
        eval_callback = WandBCallback(1, 1, eval_envs)

        model.learn(TOTAL_STEPS, callback=eval_callback)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VERBOSE = 1
USE_WANDB = 1

N_ENVS = 16
BATCH_SIZE = 16
N_STEPS = 256
TOTAL_STEPS = BATCH_SIZE * N_STEPS

observation_space = Box(-100, +100, shape=(3203,), dtype=np.float32)
action_space = Box(-100, +100, shape=(400,), dtype=np.float32)

if __name__ == "__main__":
    if USE_WANDB:
        # wandb.init(project="schmalegger-hbf", entity="schmalegg")
        sweep_id = "schmalegg/schmalegger-hbf/cz7uo6vd"
        wandb.agent(sweep_id, function=train, count=10)
    else:
        train()
