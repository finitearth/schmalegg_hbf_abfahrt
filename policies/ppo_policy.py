import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import make_proba_distribution
from torch_geometric.nn import SAGEConv, GATv2Conv, TransformerConv, global_mean_pool
from torch.nn import Linear, LazyBatchNorm1d
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch


def get_model(multi_env, config, batch_size, n_steps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pol = CustomActorCriticPolicy
    model = PPO(
        pol,
        multi_env,
        vf_coef=config.vf_coef,
        learning_rate=config.learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        clip_range=config.clip_range,
        device=device,
        gamma=config.gamma,
        verbose=1,
        policy_kwargs={"config": config}
    )

    return model


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, config=None, log_std_init=0, use_sde=False, **kwargs):
        self.config = config
        self.iterations_before_destination = config.it_b4_dest
        self.iterations_after_destination = config.it_aft_dest
        self.hidden_neurons = config.hidden_neurons
        self.output_features = config.action_vector_size
        self.n_node_features = config.n_node_features
        self.use_bn = config.use_bn
        self.normalize = config.normalize
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.features_dim = self.features_extractor.features_dim
        self.log_std_init = log_std_init
        self.action_dist = make_proba_distribution(action_space)
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        self.mlp_extractor = Extractor(self.config)

        self.value_net = ValueNet(self.hidden_neurons)
        self.policy_net = PolicyNet(self.hidden_neurons, self.output_features)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                                log_std_init=self.log_std_init)
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False, use_sde=False)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._convert_observation(obs)
        latent = self.mlp_extractor(data.x, data.edge_index_connections, data.edge_index_destinations)

        values = self.value_net(latent, data.edge_index_connections, data.edge_index_destinations, data.batch)
        latent, _ = to_dense_batch(latent, data.batch)
        mean_actions = self.policy_net(latent, data.edge_index_connections, data.edge_index_destinations)
        mean_actions = torch.flatten(mean_actions, start_dim=1)
        mean_actions = torch.hstack((mean_actions, torch.zeros(mean_actions.shape[0], 400 - mean_actions.size()[1]))).to(self.device)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._convert_observation(obs)
        latent = self.mlp_extractor(data.x, data.edge_index_connections, data.edge_index_destinations)
        values = self.value_net(latent, data.edge_index_connections, data.edge_index_destinations, data.batch)
        latent, _ = to_dense_batch(latent, data.batch)
        mean_actions = self.policy_net(latent, data.edge_index_connections, data.edge_index_destinations)
        mean_actions = torch.flatten(mean_actions, start_dim=1)
        mean_actions = torch.hstack((mean_actions, torch.zeros(mean_actions.shape[0], 400 - mean_actions.size()[1]))).to(self.device)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def _convert_observation(self, obs):
        n_batches = obs.shape[0]
        datas = []
        for i in range(n_batches):
            n_stations = int(obs[i, -1])
            n_edges = int(obs[i, -2])
            n_passenger = int(obs[i, -3])

            edge_index_connections0 = obs[i, :n_edges]
            edge_index_connections1 = obs[i, 100:100 + n_edges]
            edge_index_connections = torch.vstack((edge_index_connections0, edge_index_connections1)).long().to(self.device)

            edge_index_destinations0 = obs[i, 200:200 + n_passenger]
            edge_index_destinations1 = obs[i, 1200:1200 + n_passenger]
            edge_index_destinations = torch.vstack((edge_index_destinations0, edge_index_destinations1)).long().to(self.device)

            input_vectors = obs[i, 2200:2200 + n_stations * self.n_node_features]
            input_vectors = torch.reshape(input_vectors, (n_stations, self.n_node_features))

            data = CustomData(x=input_vectors,
                              edge_index_connections=edge_index_connections,
                              edge_index_destinations=edge_index_destinations)

            datas.append(data)

        data_loader = DataLoader(datas, batch_size=n_batches, shuffle=False)

        return next(iter(data_loader))


class Extractor(nn.Module):
    def __init__(self, config):
        super(Extractor, self).__init__()
        self.hidden_neurons = config.hidden_neurons
        self.features_dim = self.hidden_neurons
        self.latent_dim_pi = self.hidden_neurons
        self.latent_dim_vf = self.hidden_neurons

        self.conv1 = SAGEConv(config.n_node_features, config.hidden_neurons, normalize=config.normalize, bias=not config.use_bn)
        if config.use_bn: self.bn1 = LazyBatchNorm1d()
        self.conv2 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=config.normalize, bias=not config.use_bn, aggr="add")
        if config.use_bn: self.bn2 = LazyBatchNorm1d()
        self.conv3 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=config.normalize, bias=not config.use_bn)
        if config.use_bn: self.bn3 = LazyBatchNorm1d()
        self.conv4 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=config.normalize, bias=not config.use_bn)
        if config.use_bn: self.bn4 = LazyBatchNorm1d()

        self.activation = config.activation

        self.config = config

    def forward(self, x, edge_index_connections, edge_index_destinations, use_sde=False):
        x = self.conv1(x, edge_index_connections)
        x = self.activation(x)
        if self.config.use_bn: x = self.bn1(x)
        for _ in range(self.config.it_b4_dest):
            x = self.conv4(x, edge_index_connections)
            x = self.activation(x)
            if self.config.use_bn: x = self.bn2(x)

        x = self.conv2(x, edge_index_destinations)
        x = self.activation(x)
        if self.config.use_bn:  x = self.bn3(x)

        for _ in range(self.config.it_aft_dest):
            x = self.conv3(x, edge_index_connections)
            x = self.activation(x)
            if self.config.use_bn: x = self.bn4(x)

        return x


class PolicyNet(nn.Module):
    def __init__(self, hidden_neurons, output_features):
        super(PolicyNet, self).__init__()
        self.lin1 = Linear(hidden_neurons, output_features)

    def forward(self, x, edge_index_connections, edge_index_destinations):
        x = self.lin1(x)

        return x


class ValueNet(nn.Module):
    def __init__(self, hidden_neurons):
        super(ValueNet, self).__init__()
        self.lin1 = Linear(hidden_neurons, 1)

    def forward(self, x, edge_index_connections, edge_index_destinations, batch):
        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x


class CustomData(Data):
    def __init__(self, x=None, edge_index_connections=None, edge_index_destinations=None, edge_attr=None, **kwargs):
        super().__init__(x=x, edge_attr=edge_attr, **kwargs)
        self.edge_index_destinations = edge_index_destinations
        self.edge_index_connections = edge_index_connections
