import io
from PIL import Image, ImageDraw
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from enviroments import env, env2, env3, env4
from policies import ppo_policy, ppo_policy2, ppo_policy_mit_sd


def create_nx_graph(station1s, station2s):
    graph = list(zip(station1s, station2s))

    nx_graph = nx.Graph()
    for station in set(station1s):
        nx_graph.add_node(station)
    for source, target in graph:
        nx_graph.add_edge(source, target)

    graph_nx = nx.draw(nx_graph, with_labels=True)
    return graph_nx


class ConfigParams:
    def __init__(self, wandb_config=None):
        w = wandb_config is not None
        self.vf_coef =                 wandb_config.vf_coef if w            else 0.5
        self.learning_rate =           wandb_config.learning_rate if w      else 10 ** -3
        self.gamma =                   wandb_config.gamma if w              else 0.99
        self.clip_range =              wandb_config.clip_range if w         else 0.35
        self.n_node_features =         wandb_config.n_node_features if w    else 2
        self.action_vector_size =      wandb_config.action_vector_size if w else 4
        self.hidden_neurons =          wandb_config.hidden_neurons if w     else 4
        self.it_b4_dest =              wandb_config.it_b4_dest if w         else 30
        self.it_aft_dest =             wandb_config.it_aft_dest if w        else 19
        self.use_bn =                  wandb_config.use_bn if w             else True
        self.normalize =               wandb_config.normalize if w          else True
        self.log_std_init =            wandb_config.log_std_init if w       else -3.0
        self.reward_per_step = wandb_config.reward_per_step if w            else -1.0
        self.reward_reached_dest = wandb_config.reward_reached_dest if w    else 2.0

        env_str =                      wandb_config.env if w                else "env"
        envs = {"env": env, "env2": env2, "env3": env3, "env4": env4}
        self.env = envs[env_str]

        policy_str =                   wandb_config.policy if w             else "ppo_policy_with_sde"
        policies = {"ppo_policy": ppo_policy, "ppo_policy2": ppo_policy2, "ppo_policy_with_sde": ppo_policy_mit_sd}
        self.policy = policies[policy_str]

        activation_str =               wandb_config.activation if w         else "softplus"
        activations = {"tanh": torch.tanh, "relu": torch.relu, "softplus": nn.Softplus(), "None": nn.Identity()}
        self.activation = activations[activation_str]


def plot_to_image(step):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    im = Image.open(img_buf)

    d = ImageDraw.Draw(im)
    d.text((28, 36), f"Step {step}", fill=(255, 0, 0))

    im = np.array(im)

    return im
