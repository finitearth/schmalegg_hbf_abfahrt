from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import make_proba_distribution
import torch.nn as nn
import torch as th
import nets.netsgat as netsforreal

from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution, StateDependentNoiseDistribution
import gym
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

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
                if s1 != s2:
                    edge_index[0].append(s1)
                    edge_index[1].append(s2)
      # edge_index = [range(5), range(5)]
        self.net = netsforreal.CustomNet(edge_index)
        # self.value_net = self.mlp_extractor.value_net
        # self.action_net = self.mlp_extractor.policy_net

        # Disable orthogonal initialization
        self.ortho_init = True
        #self.value_net = netsforreal.ValueNet(edge_index)
        #self.action_net = netsforreal.PolicyNet(edge_index)
        # self.log_std = 5
        # self.log_std = th.nn.Parameter(th.tensor([0.5]), requires_grad=True)
        # self.log_std2 = th.nn.Parameter(th.tensor([0.5]), requires_grad=True)
        # self.optimizer = th.optim.Adam(self.parameters())

    # def _build_mlp_extractor(self) -> None:
    #     self.mlp_extractor = self.net

    # def _build(self, lr_schedule) -> None:
    #     edge_index = [[], []]
    #     for s1 in range(5):
    #         for s2 in range(5):
    #             edge_index[0].append(s1)
    #             edge_index[1].append(s2)
    #
    #     self.mlp_extractor = netsforreal.CustomNet(edge_index)

    def forward(self, x):
      #  x = self.extract_features(x)
      #  print(x)
        # Evaluate the values for the given observations
        actions, values = self.net(x)
        distribution = self._get_action_dist_from_latent(actions)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def _get_action_dist_from_latent(self, actions):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
#        mean_actions, _ =  self.net(actions)

        if True:#True:#isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(actions, self.log_std)
        elif False:#isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(mean_actions=actions, log_std=self.log_std)#action_logits=actions)#mean_actions)
        elif False:#isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(mean_actions=actions, log_std=self.log_std)#action_logits=actions)
        elif False:#isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(actions, self.log_std, x)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = obs#self.extract_features(obs)
        latent_pi, latent_vf = self.net(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = latent_vf# self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def predict(self, obs, state=None, mask=None, deterministic=False):
        obs = th.as_tensor(obs)
        actions, _ = self.net(obs)
        return actions

