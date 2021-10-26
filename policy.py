from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import make_proba_distribution
import torch.nn as nn
import torch as th
import netsforreal
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution, StateDependentNoiseDistribution
import numpy as np
import gym
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.torch_layers import FlattenExtractor


# class Policy(policies.ActorCriticPolicy):
#     def __init__(self, observation_space=IDK, action_space=IDK):
#         super(Policy, self).__init__(observation_space, action_space)
#         self.net = nets.Net()
#
#     def forward(self, x):
#         x = self.net(x)
#
#         return x


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        edge_index = [[], []]
        for s1 in range(5):
            for s2 in range(5):
                edge_index[0].append(s1)
                edge_index[1].append(s2)
        self.net = netsforreal.CustomNet(edge_index)
        self.value_net = self.mlp_extractor.value_net
        self.action_net = self.mlp_extractor.policy_net

        # Disable orthogonal initialization
        self.ortho_init = True
        self.value_net = netsforreal.ValueNet(edge_index)
        self.action_net = netsforreal.PolicyNet(edge_index)
        # self.log_std = 5
        self.log_std = th.nn.Parameter(th.tensor([0.5] * 20, requires_grad=True))
        self.optimizer = th.optim.Adam(self.parameters())

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = self.net

    def _build(self, lr_schedule) -> None:
        edge_index = [[], []]
        for s1 in range(5):
            for s2 in range(5):
                edge_index[0].append(s1)
                edge_index[1].append(s2)

        self.mlp_extractor = netsforreal.CustomNet(edge_index)
