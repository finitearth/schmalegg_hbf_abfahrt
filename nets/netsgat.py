import torch
import torch.nn as nn
from stable_baselines3.common.distributions import make_proba_distribution
from torch_geometric.nn import SAGEConv
from torch.nn import Linear, LazyBatchNorm1d
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch


NODE_FEATURES = 5
OUTPUT_FEATURES = 2
HIDDEN_NEURONS = 8

ITERATIONS_BEFORE_DESTINATION = 1
ITERATIONS_AFTER_DESTINATION = 2


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.features_dim = HIDDEN_NEURONS
        self.latent_dim_pi = HIDDEN_NEURONS
        self.latent_dim_vf = HIDDEN_NEURONS

        self.conv1 = SAGEConv(NODE_FEATURES, HIDDEN_NEURONS, normalize=True, bias=False)
        self.bn1 = LazyBatchNorm1d()
        self.conv2 = SAGEConv(HIDDEN_NEURONS, HIDDEN_NEURONS, normalize=True, bias=False)
        self.bn2 = LazyBatchNorm1d()
        self.conv3 = SAGEConv(HIDDEN_NEURONS, HIDDEN_NEURONS, normalize=True, bias=False)
        self.bn3 = LazyBatchNorm1d()
        self.conv4 = SAGEConv(HIDDEN_NEURONS, HIDDEN_NEURONS, normalize=True, bias=False)
        self.bn4 = LazyBatchNorm1d()

    def forward(self, x, edge_index_connections, edge_index_destinations):
        x = self.conv1(x, edge_index_connections)
        x = F.relu(x)
        x = self.bn1(x)
        for _ in range(ITERATIONS_BEFORE_DESTINATION):
            x = self.conv4(x, edge_index_connections)
            x = F.relu(x)
            x = self.bn2(x)

        x = self.conv2(x, edge_index_destinations)
        x = F.relu(x)
        x = self.bn3(x)

        for _ in range(ITERATIONS_AFTER_DESTINATION):
            x = self.conv3(x, edge_index_connections)
            x = F.relu(x)
            x = self.bn4(x)

        return x


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.lin1 = Linear(HIDDEN_NEURONS, OUTPUT_FEATURES)

    def forward(self, x, edge_index_connections, edge_index_destinations):
        x = self.lin1(x)
        x = torch.flatten(x, start_dim=1)

        return x


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        # self.lin1 = Linear(HIDDEN_NEURONS, HIDDEN_NEURONS // 4)
        # self.bn = LazyBatchNorm1d()
        self.lin2 = Linear(HIDDEN_NEURONS, 1, bias=False)# // 4, 1)

    def forward(self, x, edge_index_connections, edge_index_destinations):
        # x = self.lin1(x)
        # x = F.relu(x)
        # x = self.bn(x)
        x = torch.sum(x, 1, keepdim=False)  # davor war hier ein mean lul
        x = self.lin2(x)

        return x


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, log_std_init=0, use_sde=False, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.features_extractor = Extractor()
        self.features_dim = self.features_extractor.features_dim
        self.log_std_init = log_std_init
        self.action_dist = make_proba_distribution(action_space)
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        self.mlp_extractor = Extractor()
        self.value_net = ValueNet()
        self.policy_net = PolicyNet()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi,
                                                                                log_std_init=self.log_std_init)
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._convert_observation(obs)
        latent = self.mlp_extractor(data.x, data.edge_index_connections, data.edge_index_destinations)
        latent, _ = to_dense_batch(latent, data.batch)

        values = self.value_net(latent, data.edge_index_connections, data.edge_index_destinations)

        mean_actions = self.policy_net(latent, data.edge_index_connections, data.edge_index_destinations)
        mean_actions = torch.hstack((mean_actions, torch.zeros(mean_actions.shape[0], 400 - mean_actions.size()[1])))
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self._convert_observation(obs)
        latent = self.mlp_extractor(data.x, data.edge_index_connections, data.edge_index_destinations)
        latent, _ = to_dense_batch(latent, data.batch)
        values = self.value_net(latent, data.edge_index_connections, data.edge_index_destinations)

        mean_actions = self.policy_net(latent, data.edge_index_connections, data.edge_index_destinations)
        mean_actions = torch.hstack((mean_actions, torch.zeros(mean_actions.shape[0], 400 - mean_actions.size()[1])))
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def _convert_observation(self, obs):# torch.Batch):
        n_batches = obs.shape[0]
        datas = []
        for i in range(n_batches):
            n_stations = int(obs[i, -1])
            n_edges = int(obs[i, -2])
            n_passenger = int(obs[i, -3])

            edge_index_connections0 = obs[i, :n_edges]
            edge_index_connections1 = obs[i, 100:100 + n_edges]
            edge_index_connections = torch.vstack((edge_index_connections0, edge_index_connections1)).long()

            edge_index_destinations0 = obs[i, 200:200 + n_passenger]
            edge_index_destinations1 = obs[i, 1200:1200 + n_passenger]
            edge_index_destinations = torch.vstack((edge_index_destinations0, edge_index_destinations1)).long()

            input_vectors = obs[i, 2200:2200 + n_stations * NODE_FEATURES]
            input_vectors = torch.reshape(input_vectors, (n_stations, NODE_FEATURES))

            data = CustomData(x=input_vectors,
                              edge_index_connections=edge_index_connections,
                              edge_index_destinations=edge_index_destinations)

            datas.append(data)

        data_loader = DataLoader(datas, batch_size=n_batches, shuffle=False)  # , num_workers=1)

        return next(iter(data_loader))


class CustomData(Data):
    def __init__(self, x=None, edge_index_connections=None, edge_index_destinations=None, edge_attr=None, **kwargs):
        super().__init__(x=x, edge_attr=edge_attr, **kwargs)
        self.edge_index_destinations = edge_index_destinations
        self.edge_index_connections = edge_index_connections

#
# class CustomObservationBuffer(BaseBuffer):
#     def __init__(self, buffer_size, observation_space, action_space):
#         super().__init__(buffer_size, observation_space, action_space)
