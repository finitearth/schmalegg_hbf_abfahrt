import io
import cv2
from PIL import ImageDraw, Image

import utils
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
import time
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from env import AbfahrtEnv

plt.switch_backend('agg')


def get_callbacks(envs=None, use_wandb=False, config=None):
    callback_factor = 4
    eval_callback = EvalCallback(envs, eval_freq=config.n_steps*callback_factor, n_eval_episodes=12)
    render_callback = RenderCallback(config.n_steps*callback_factor, envs, use_wandb, config=config)

    return CustomCallBacklist([eval_callback, render_callback])


class CustomCallBacklist(CallbackList):
    def __init__(self, callbacks):
        super(CustomCallBacklist, self).__init__(callbacks)

    def _on_step(self) -> bool:
        t0 = time.perf_counter()
        continue_training = True
        for callback in self.callbacks:
            continue_training = callback.on_step() and continue_training
        dt = time.perf_counter() - t0
        if dt > 1: print(f"eval took {round(dt, 1)} s")
        return continue_training


class RenderCallback(BaseCallback):
    def __init__(self, eval_freq, env, use_wandb=False, config=None):
        super(RenderCallback, self).__init__(verbose=1)
        self.eval_freq = eval_freq
        self.use_wandb = use_wandb
        self.config = config

        self.env = env
        self.frames = []

    def _on_step(self):
        try: 
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                _, _, success = _evaluate_policy(self.model, 1, render_fct=self.render_frame, prints=False, config=self.config)
                self.render_video()
                if success < 50: return False
            return True
        except Exception as e:
            print(e)
            return True

    def render_video(self):
        frames = self.frames
        frames.append(np.ones_like(frames[0]))
        frames_arr = np.asarray(frames)
        if self.use_wandb:
            frames_arr = np.transpose(frames_arr, (0, 3, 1, 2))
            video = wandb.Video(frames_arr, fps=1,
                                caption=f"Iteration {self.n_calls // self.eval_freq}", format="mp4")
            wandb.log({"eval_video": video})
        else:
            out = cv2.VideoWriter(f'videos/output{self.n_calls}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (640, 480))
            for frame in frames_arr: out.write(frame.astype('uint8'))
            out.release()
        self.frames = []

    def render_frame(self, eval_env, step):
        routes = eval_env.routes
        trains = eval_env.trains
        stations = eval_env.stations
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

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')

        im = Image.open(img_buf)

        black = (0, 0, 0)

        d = ImageDraw.Draw(im)
        d.text((28, 36), f"Iteration {self.n_calls}\nStep {step}", fill=black)

        point_map = {0: (333, 323), 1: (533, 145), 2: (355, 100), 3: (120, 364), 4: (177, 266), 5: (365, 214),
                     6: (470, 409)}
        for s in stations:
            s_ = int(s)
            if s.vector is not None:
                p0, p1 = int(s.vector[0]*50), int(s.vector[1]*50)
                d, im = utils.draw_arrow(im, (point_map[s_][0] - 50, point_map[s_][1]),
                                         (point_map[s_][0] - 50 + p0, point_map[s_][1] + p1), color=(0, 0, 255),
                                         thickness=3)
                p0, p1 = int(s.vector[2] * 50), int(s.vector[3] * 50)
                d, im = utils.draw_arrow(im, (point_map[s_][0] - 100, point_map[s_][1]),
                                         (point_map[s_][0] - 100 + p0, point_map[s_][1] + p1), color=(255, 0, 0),
                                         thickness=3)
            if s.passengers:
                d, im = utils.draw_arrow(im, point_map[s_], (point_map[s_][0], point_map[s_][1] + 25))
                for i, p in enumerate(s.passengers, start=2):
                    d.text((point_map[s_][0]-50, point_map[s_][1]+i*12), f"In Station -> {int(p.destination)}", fill=black)

        for t in trains:
            if t.passengers:
                s_ = int(t.station)
                d, im = utils.draw_arrow(im, point_map[s_], (point_map[s_][0], point_map[s_][1]+25))
                for i, p in enumerate(t.passengers, start=2):
                    d.text((point_map[s_][0]-50, point_map[s_][1]+i*12), f"In Train -> {int(p.destination)}", fill=black)
        im = im.convert('RGB')
        self.frames.append(np.array(im))
        plt.close()


def _evaluate_policy(model, n_eval_episodes, render_fct=None, prints=True, config=None):
    eval_env = AbfahrtEnv(mode="render", config=config)
    eval_env.reset()
    episode_rewards = []
    episode_lengths = []
    for _ in range(n_eval_episodes):
        observation = eval_env.reset()
        current_reward = 0
        steps_taken = 0
        for i in range(200):
            if render_fct is not None: render_fct(eval_env, i)
            observation = torch.Tensor(observation).unsqueeze(0)

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
