import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import time
import numpy as np
import torch


def get_callbacks(logger=None, envs=None, n_steps=None, use_wandb=False):
    eval_callback = EvalCallBack(n_steps, envs, use_wandb)

    training_callback = WandBTrainingCallBack(n_steps, logger, use_wandb)

    return CallbackList([eval_callback, training_callback])


class WandBTrainingCallBack(BaseCallback):
    def __init__(self, eval_freq, logger, use_wandb):
        super(WandBTrainingCallBack, self).__init__(verbose=1)
        self.eval_freq = eval_freq
        self.logger = logger
        self.use_wandb = use_wandb

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.use_wandb:

            wandb.log({"value_loss": self.logger.name_to_value["train/value_loss"]})
            wandb.log({"gradient_loss": self.logger.name_to_value["train/policy_gradient_loss"]})
            wandb.log({"entropy_loss": self.logger.name_to_value["train/entropy_loss"]})


class EvalCallBack(BaseCallback):
    def __init__(self, eval_freq, eval_env=None, use_wandb=False):
        super(EvalCallBack, self).__init__(verbose=1)
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = 16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb

    def init_callback(self, model):
        self.model = model

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []
            mean_reward, mean_length, perc_successfull = _evaluate_policy(self.model,
                                                                          self.eval_env,
                                                                          self.n_eval_episodes,
                                                                          self.device)
            if self.use_wandb:
                wandb.log({"episode_lenghts": mean_length})
                wandb.log({"succesfull_episodes %": perc_successfull})

        return True


def _evaluate_policy(model, eval_env, n_eval_episodes, device):
    episode_rewards = []
    episode_lengths = []
    t0 = time.perf_counter()
    for _ in range(n_eval_episodes):
        observation = eval_env.reset()
        current_reward = 0
        steps_taken = 0
        for i in range(200):
            observation = torch.tensor(observation).float()

            action, _, _ = model.policy.forward(observation, deterministic=False)

            action = action.detach().numpy()
            observation, reward, done, info = eval_env.step(action)
            current_reward += reward

            if done: break
            steps_taken += 1

        episode_rewards.append(current_reward)
        episode_lengths.append(steps_taken)
    time_taken = time.perf_counter() - t0
    print(f"======== EVALUATION ========= it took: {time_taken}")
    perc_successfull = np.count_nonzero(np.asarray(episode_lengths) < 200) / n_eval_episodes * 100

    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    best_length = np.min(episode_lengths)
    print(f"reached goal: {perc_successfull} %")
    print(f"mean_length: {mean_length}")
    print(f"best_length: {best_length}")

    return mean_reward, mean_length, perc_successfull
