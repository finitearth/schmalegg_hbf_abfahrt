from functools import partial

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch_geometric.data import Data
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import add_self_loops, to_dense_batch

import utils


def get_model(multi_env, value_net, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pol = CustomActorCriticPolicy
    model = PPO(
        pol,
        multi_env,
        vf_coef=config.vf_coef,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_steps=config.n_steps,
        clip_range=config.clip_range,
        device=device,
        gamma=config.gamma,
        verbose=20,
        policy_kwargs={"config": config,
                       "value_net": value_net}
    )

    return model


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, value_net, config=None, use_sde=False, **kwargs):
        self.config = config
        self.iterations_before_destination = config.it_b4_dest
        self.iterations_after_destination = config.it_aft_dest
        self.hidden_neurons = config.hidden_neurons
        self.output_features = config.action_vector_size
        self.n_node_features = config.n_node_features

        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.value_net = value_net
        self.features_dim = self.features_extractor.features_dim
        self.log_std_init = config.log_std_init
        self.action_dist = DiagGaussianDistribution(int(np.prod(action_space.shape)))
        self._build(lr_schedule)

    def _build(self, lr_schedule):
        self.policy_net = PolicyNet(self.config)
        _, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.config.action_vector_size,
                                                                  log_std_init=self.log_std_init)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False, use_sde=False, requires_conversion=True):
        x, eic, eid, eit, batch = utils.convert_observation(obs, self.config)
        values = self.value_net(x, eic, eid, eit, batch)
        mean_actions = self.policy_net(x, eic, eid, eit, batch)
        mean_actions, _ = to_dense_batch(mean_actions, batch)
        mean_actions = torch.flatten(mean_actions, start_dim=1)
        mean_actions = torch.hstack((mean_actions, torch.zeros(mean_actions.shape[0], 50_000 - mean_actions.size()[1])))
        if torch.isnan(self.log_std).any():
            log_std_torch =  torch.tensor(self.log_std_init, dtype=torch.float)
            self.log_std = torch.nn.parameter.Parameter(torch.where(torch.isnan(self.log_std), log_std_torch, self.log_std))
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs

    def predict(self, observation, state=None, mask=None, deterministic=False):
        x, eic, eid, eit, batch = observation# utils.convert_observation(observation, self.config)
        # if isinstance(x, np.ndarray): x = torch.Tensor(x)
        with torch.no_grad():
            action = self.policy_net(x, eic, eid, eit, batch)
            # action = action.numpy()
        return action
        # action, _, _ = self.forward(observation, deterministic)
        # action = action.cpu().detach().numpy()
        # return action, state

    # def get_action(self, observation):
    #

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        x, eic, eid, eit, batch = utils.convert_observation(obs, self.config)
        with torch.no_grad():
            values = self.value_net(x, eic, eid, eit,  batch)
            latent, _ = to_dense_batch(x, batch)
            mean_actions = self.policy_net(latent, eic, eid, eit)
        mean_actions = torch.flatten(mean_actions, start_dim=1)
        mean_actions = torch.hstack(
            (mean_actions, torch.zeros(mean_actions.shape[0], 50_000 - mean_actions.size()[1]).to(self.device))).to(
            self.device)
        if torch.isnan(self.log_std).any():
            log_std_torch = torch.tensor(self.log_std_init, dtype=torch.float)
            self.log_std = torch.nn.parameter.Parameter(
                torch.where(torch.isnan(self.log_std), log_std_torch, self.log_std))
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()


class PolicyNet(nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        self.config = config
        hidden_neurons = config.hidden_neurons
        convclass = config.conv
        self.conv1 = convclass(config.n_node_features, config.hidden_neurons, aggr=config.aggr_con)
        self.conv2 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv3 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_dest)
        self.conv4 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv5 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.lins = [Linear(hidden_neurons, hidden_neurons) for _ in range(config.n_lin_policy)]
        self.lin1 = Linear(hidden_neurons, config.action_vector_size)

    def forward(self, x, edge_index_connections, edge_index_destinations, edge_index_trains, batch):
        x = self.conv1(x, edge_index_connections)
        # x = self.activation(x)
        x = self.conv2(x, edge_index_trains)
        for _ in range(self.config.it_b4_dest):
            x = self.conv3(x, edge_index_connections)
            # x = self.activation(x)  #
        x = self.conv4(x, edge_index_destinations)
        # x = self.activation(x)

        for _ in range(self.config.it_aft_dest):
            x = self.conv5(x, edge_index_connections)
            # x = self.activation(x)
        # x, _ =
        for lin in self.lins:
            x = lin(x)
        x = self.lin1(x)
        return x


