import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple
from torch_geometric.utils import to_dense_batch
from gym.spaces.box import Box

import utils


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
     A class that implements an Actor-Critic algorithm for the AbfahrtEnviroment implementing graph neural networks.

    Args:
        observation_space (gym.spaces.Space):
            Observation space of the environment.
        action_space (gym.spaces.Space):
            Action space of the environment.
        hidden_neurons (int):
            Number of neurons in the hidden layers.
        **kwargs (dict):
            Extra parameters.

    Attributes:
        hidden_neurons (int):
            Number of neurons in the hidden layers.
        extractor (nn.Module):
            The feature extractor.
        policy_net (nn.Module):
            The policy network.
        value_net (nn.Module):
            The value network.
        action_dist (ActionDistribution):
            A action distribution.

    """
    def __init__(self, observation_space: Box, action_space: Box, hidden_neurons: int, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, **kwargs)

        self.action_dist = make_proba_distribution(action_space)
        self.extractor = Extractor(hidden_neurons)
        self.policy_net = PolicyNet(hidden_neurons)
        self.value_net = ValueNet(hidden_neurons)

    def forward(self, obs: torch.Tensor, deterministic: bool = False)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes actions, values and log probabilities given a batch of observations.

        Parameters
        ----------
        obs : torch.Tensor
            A batch of observations.
        deterministic : bool
            Whether to use stochastic or deterministic actions. Stochastic actions are valueable for exploration.

        Returns
        -------
        actions (torch.Tensor)
            The actions the agent decided to take.
        values (torch.Tensor)
            The value estimation for the current state
        torch.Tensor
            A batch of log probabilities of the actions.
        """
        data = utils.convert_observation(obs)
        latent = self.mlp_extractor(data.x, data.edge_index_connections, data.edge_index_destinations)

        values = self.value_net(latent, data.edge_index_connections, data.edge_index_destinations, data.batch)
        # torch geometric treats batches of graphs as one supergraph, pytorch however needs dense batches
        latent, _ = to_dense_batch(latent, data.batch)

        mean_actions = self.policy_net(latent, data.edge_index_connections, data.edge_index_destinations)
        distribution = self.action_dist.proba_distribution(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs


class Extractor(nn.Module):
    """
        A class defining the extractor part of the model.

        Attributes:
            conv1 (SAGEConv): The first convolution layer. Passing messages to connected stations (nodes).
            conv2 (SAGEConv): The second convolution layer. Passing messages from destination stations
                              to current stations of the passengers.
            conv3 (SAGEConv): The third convolution layer. Passing messages to adjacent stations (nodes).

        Methods:
            forward(x, edge_index_connections, edge_index_destinations):
                Returns the output of the extractor.
    """

    def __init__(self, hidden_neurons: int):
        super(Extractor, self).__init__()

        self.conv1 = SAGEConv(8, hidden_neurons, normalize=True)
        self.conv2 = SAGEConv(hidden_neurons, hidden_neurons, normalize=True)
        self.conv3 = SAGEConv(hidden_neurons, hidden_neurons, normalize=True)

    def forward(self, x: torch.Tensor, edge_index_connections: torch.Tensor, edge_index_destinations: torch.Tensor) \
            -> torch.Tensor:
        """
            Forward method of the extractor class.

            Parameters:
            x (torch.tensor): The input tensor.
            edge_index_connections (torch.tensor): edge tensor, defining connected stations
            edge_index_destinations (torch.tensor): edge tensor, abstracting the current - destination relationships

            Returns:
            x (torch.tensor): The output tensor.
        """
        x = self.conv1(x, edge_index_connections)
        x = torch.tanh(x)

        x = self.conv2(x, edge_index_destinations)
        x = torch.tanh(x)

        x = self.conv3(x, edge_index_connections)
        x = torch.tanh(x)

        return x


class PolicyNet(nn.Module):
    """
    A class defining the policy head of the agent.

    Attributes:
        lin1 (torch.nn.Linear): A linear layer with input size of hidden_neurons and output size of output_features.

    """
    def __init__(self, hidden_neurons: int):
        super(PolicyNet, self).__init__()
        self.lin1 = nn.Linear(hidden_neurons, 4)

    def forward(self, x: torch.Tensor, edge_index_connections: torch.Tensor, edge_index_destinations: torch.Tensor)\
            -> torch.Tensor:
        """
            Forward method of the PolicyNet class.

            Parameters:
            x (torch.tensor): The input tensor.
            edge_index_connections (torch.tensor): edge tensor, defining connected stations
            edge_index_destinations (torch.tensor): edge tensor, abstracting the current - destination relationships

            Returns:
            x (torch.tensor): The output tensor.
        """
        x = self.lin1(x)

        return x


class ValueNet(nn.Module):
    """
        A class defining the value head of the agent.

        Parameters:
            hidden_neurons (int): The number of neurons in the hidden layer.

        Attributes:
            lin1 (torch.nn.Linear): A linear layer.
    """
    def __init__(self, hidden_neurons):
        super(ValueNet, self).__init__()
        self.lin1 = nn.Linear(hidden_neurons, 1)

    def forward(self, x: torch.Tensor, edge_index_connections: torch.Tensor, edge_index_destinations: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
                   Forward method of the ValueNet class. Applies global mean pooling to add a super node,
                   representing the entire graph
             Args:
                 x (torch.Tensor): The input data.
                 edge_index_connections (torch.tensor): edge tensor, defining connected stations
                 edge_index_destinations (torch.tensor): edge tensor, abstracting the current-destination relationships
                 batch (torch.Tensor): The batch vector, associating each node in the super graph to its batch
             Returns:
                 x (torch.Tensor): The output of the ValueNet.
        """
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x
