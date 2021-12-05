import networkx as nx
import torch
from torch import nn

from enviroments import env, env2, env3, env4
from policies import ppo_policy, ppo_policy2


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

        env_str =                      wandb_config.env if w                else "env3"
        envs = {"env": env, "env2": env2, "env3": env3, "env4": env4}
        self.env = envs[env_str]

        policy_str =                   wandb_config.policy if w             else "ppo_policy"
        policies = {"ppo_policy": ppo_policy, "ppo_policy2": ppo_policy2}
        self.policy = policies[policy_str]

        activation_str =               wandb_config.activation if w         else "tanh"
        activations = {"tanh": torch.tanh, "relu": torch.relu, "None": nn.Identity()}
        self.activation = activations[activation_str]



