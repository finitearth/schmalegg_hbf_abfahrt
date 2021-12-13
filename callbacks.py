from matplotlib.backends.backend_template import FigureCanvas

import utils
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import time
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def get_callbacks(logger=None, envs=None, n_steps=None, use_wandb=False):
    eval_callback = EvalCallBack(n_steps, envs, use_wandb)
    training_callback = WandBTrainingCallBack(n_steps, logger, use_wandb)
    render_callback = RenderCallback(n_steps*1, envs, use_wandb)
    return CustomCallBacklist([eval_callback, training_callback, render_callback])


class CustomCallBacklist(CallbackList):
    def __init__(self, list):
        super(CustomCallBacklist, self).__init__(list)

    def _on_step(self) -> bool:
        t0 = time.perf_counter()
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        dt = time.perf_counter() - t0
        if dt > 1: print(f"eval took {round(dt, 1)} s")
        return continue_training


class WandBTrainingCallBack(BaseCallback):
    def __init__(self, eval_freq, logger, use_wandb):
        super(WandBTrainingCallBack, self).__init__(verbose=1)
        self.eval_freq = eval_freq
        self.logger = logger
        self.use_wandb = use_wandb

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.use_wandb:
            wandb.log(dict(self.logger.name_to_value))


class EvalCallBack(BaseCallback):
    def __init__(self, eval_freq, eval_env=None, use_wandb=False):
        super(EvalCallBack, self).__init__(verbose=1)
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = 16
        self.use_wandb = use_wandb

    def init_callback(self, model):
        self.model = model

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []
            mean_reward, mean_length, perc_successfull = _evaluate_policy(self.model,
                                                                          self.eval_env,
                                                                          self.n_eval_episodes)
            if self.use_wandb:
                wandb.log({"episode_lenghts": mean_length})
                wandb.log({"succesfull_episodes %": perc_successfull})

        return True


def _evaluate_policy(model, eval_env, n_eval_episodes, render_fct=None, prints=True):
    episode_rewards = []
    episode_lengths = []
    for _ in range(n_eval_episodes):
        observation = eval_env.reset()
        current_reward = 0
        steps_taken = 0
        for i in range(200):
            if render_fct is not None: render_fct(eval_env.routes, eval_env.trains, i)
            observation = torch.tensor([observation]).float()

            action, _, _ = model.policy.forward(observation, deterministic=False)

            action = action.cpu().detach().numpy()
            action = action[0]
            observation, reward, done, info = eval_env.step(action)
            current_reward += reward

            if done: break
            steps_taken += 1

        episode_rewards.append(current_reward)
        episode_lengths.append(steps_taken)

    if prints: print(f"======== EVALUATION =========")
    perc_successfull = np.count_nonzero(np.asarray(episode_lengths) < 200) / n_eval_episodes * 100

    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    best_length = np.min(episode_lengths)
    if prints: print(f"reached goal: {perc_successfull} %")
    if prints: print(f"mean_length: {mean_length}")
    if prints: print(f"best_length: {best_length}")

    return mean_reward, mean_length, perc_successfull


class RenderCallback(BaseCallback):
    def __init__(self, eval_freq, env, use_wandb=False):
        super(RenderCallback, self).__init__(verbose=1)
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb

        self.env = env
        self.frames = []

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            _evaluate_policy(self.model, self.env, 1, render_fct=self.render_frame, prints=False)
            self.render_video()
        return True

    def render_video(self):
        frames = self.frames
        frames.append(np.ones_like(frames[0]))
        frames_arr = np.asarray(frames)
        frames_arr = np.transpose(frames_arr, (0, 3, 1, 2))
        if self.use_wandb:
            wandb.log({"eval_video": wandb.Video(frames_arr, fps=1, caption=f"Iteration {self.n_calls//self.eval_freq}", format="mp4")})
        self.frames = []

    def render_frame(self, routes, trains, step):
        graph = list(zip(routes[0], routes[1]))
        nx_graph = nx.Graph()
        for node in set(routes[0]):
            nx_graph.add_node(node)
        train_colors = ["red", "green", "yellow", "pink", "grey"]
        train_to_colors = {}
        for i, t in enumerate(trains):
            train_to_colors[t] = train_colors[i]

        for source, target in graph:
            nx_graph.add_edge(source, target)

        color_map = ["blue"] * len(set(routes[0]))

        for t in trains:
            station = int(t.station)
            color_map[station] = train_to_colors[t]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        nx.draw_kamada_kawai(nx_graph, with_labels=True, ax=ax, node_color=color_map)

        frame = utils.plot_to_image(step)
        plt.close()
        self.frames.append(frame)
