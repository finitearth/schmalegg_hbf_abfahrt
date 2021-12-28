import cv2

import wandb
from PIL import Image, ImageDraw
import networkx as nx
import numpy as np
import torch
from stable_baselines3.common.logger import Logger, make_output_format, KVWriter
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, GraphConv, GCN2Conv
import env
from policies import ppo_policy


class ConfigParams:
    def __init__(self, wandb_config=None):
        w = wandb_config is not None
        self.vf_coef =                 wandb_config.vf_coef if w            else 0.5
        self.learning_rate =           wandb_config.learning_rate if w      else 10**-6
        self.gamma =                   wandb_config.gamma if w              else 0.9999
        self.clip_range =              wandb_config.clip_range if w         else 0.2
        self.n_node_features =         wandb_config.n_node_features if w    else 4
        self.action_vector_size =      wandb_config.action_vector_size if w else 8
        self.hidden_neurons =          wandb_config.hidden_neurons if w     else 32
        self.it_b4_dest =              wandb_config.it_b4_dest if w         else 20
        self.it_aft_dest =             wandb_config.it_aft_dest if w        else 20
        self.normalize =               wandb_config.normalize if w          else True
        self.log_std_init =            wandb_config.log_std_init if w       else -2

        self.reward_per_step =         wandb_config.reward_per_step if w    else 0# -.05
        self.reward_reached_dest =     wandb_config.reward_reached_dest if w    else 1
        self.reward_step_closer =      wandb_config.reward_step_closer if w else .1

        self.aggr_dest =               wandb_config.aggr_dest if w          else "add"
        self.aggr_con =                wandb_config.aggr_con if w           else "mean"

        self.n_envs =                  8
        self.batch_size = wandb_config.batch_size if w else 16
        n_steps = wandb_config.n_steps if w else 48
        self.n_steps = n_steps + self.batch_size - (n_steps % self.batch_size) # such that batch_size is a factor of n_steps
        self.total_steps = self.n_steps * self.n_envs * self.batch_size * 128

        self.n_lin_extr =      wandb_config.n_lin_extr if w else 3
        self.n_lin_policy =    wandb_config.n_lin_policy if w else 1
        self.n_lin_value =      wandb_config.n_lin_value if w else 2

        self.range_inputvec = wandb_config.range_inputvec if w else .02

        self.env = env
        self.policy = ppo_policy

        activation_str =               wandb_config.activation if w         else "None"
        activations = {"tanh": torch.tanh, "relu": torch.relu, "None": nn.Identity()}
        self.activation = activations[activation_str]

        conv_str =                     wandb_config.conv if w               else "GCNConv"#"GCNConv"
        convs = {"GCNConv": GCNConv, "SAGEConv": SAGEConv, "GraphConv": GraphConv,
                 "GATConv": GATConv, "GATv2Conv": GATv2Conv}
        self.conv = convs[conv_str]


class CustomLogger(Logger):
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb
        super(CustomLogger, self).__init__("",  [make_output_format("stdout", "./logs", "")])

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)
        if self.use_wandb: wandb.log(self.name_to_value)
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()


def create_nx_graph(station1s, station2s):
    graph = list(zip(station1s, station2s))
    nx_graph = nx.Graph()
    for station in set(station1s):
        nx_graph.add_node(station)
    for source, target in graph:
        nx_graph.add_edge(source, target)

    graph_nx = nx.draw(nx_graph, with_labels=True)
    return graph_nx


def draw_arrow(im, p0, p1, thickness=1, color=(0, 0, 0)):
    na = np.array(im)
    na = cv2.arrowedLine(na, p0, p1, color, thickness)
    im = Image.fromarray(na)
    return ImageDraw.Draw(im), im
