import collections
import warnings
import torch
import torch.nn as nn
from stable_baselines3.common.distributions import make_proba_distribution, StateDependentNoiseDistribution, \
    DiagGaussianDistribution, CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution
from torch_geometric.nn import SAGEConv, GCNConv, TopKPooling, LayerNorm
from torch.nn import Linear
from stable_baselines3.common.policies import ActorCriticPolicy
import gym
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch as th

NODE_FEATURES = 10
OUTPUT_FEATURES = 4
HIDDEN_NEURONS = 16


class Extractor(nn.Module):
    def __init__(self, edge_index):
        super(Extractor, self).__init__()
        self.edge_index = edge_index
        self.features_dim = HIDDEN_NEURONS
        self.latent_dim_pi = HIDDEN_NEURONS
        self.latent_dim_vf = HIDDEN_NEURONS

        self.conv1 = SAGEConv(10, HIDDEN_NEURONS)
        self.norm = LayerNorm(HIDDEN_NEURONS)

    def forward(self, x):
        x = self.conv1(x, self.edge_index)
       # x = self.norm(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, edge_index, *args, **kwargs):
        super(PolicyNet, self).__init__()
        self.edge_index = edge_index#torch.tensor(edge_index)

        self.conv2 = GCNConv(HIDDEN_NEURONS, OUTPUT_FEATURES)
        self.norm = LayerNorm(HIDDEN_NEURONS)

    def forward(self, x, use_sde=False):
        x = self.conv2(x, self.edge_index)
       # x = self.norm(x)
        x = torch.flatten(x, start_dim=1)

        return x


class ValueNet(nn.Module):
    def __init__(self, edge_index, *args, **kwargs):
        super(ValueNet, self).__init__()
        self.edge_index = edge_index
        self.pool = TopKPooling(HIDDEN_NEURONS)
        self.linear = Linear(HIDDEN_NEURONS, 1)

    def forward(self, x, use_sde=False):

        # x = global_mean_pool(x, torch.zeros_like(x, dtype=torch.int64))
        x = torch.mean(x, 1, keepdim=False)
        x = self.linear(x)*10  # , self.edge_index)

        return x


# class CustomNet(nn.Module):
#     def __init__(self, edge_index, *args, **kwargs):
#         super(CustomNet, self).__init__()
#         self.edge_index = torch.tensor(edge_index)
#         self.value_net = ValueNet(self.edge_index)
#         self.policy_net = PolicyNet(self.edge_index)
#         self.latent_dim_pi = HIDDEN_NEURONS
#         self.latent_dim_vf = HIDDEN_NEURONS
#
#     def forward(self, x, use_sde=False):
#         x_v = self.value_net(x)
#         x_p = self.policy_net(x)
#
#         return x_p, x_v


class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            *args,
            **kwargs,
    ):
        features_extractor_class = Extractor
        features_extractor_kwargs = {}

        edge_index = [[], []]
        for s1 in range(5):
            for s2 in range(5):
                edge_index[0].append(s1)
                edge_index[1].append(s2)

        self.edge_index = th.tensor(edge_index)

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

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
        # Disable orthogonal initialization
        self.ortho_init = False

        #  self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.edge_index)
        self.features_dim = self.features_extractor.features_dim

       # self.action_net = PolicyNet(self.edge_index)

        #  self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist,
                          StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = Extractor(self.edge_index)

    def _build(self, lr_schedule) -> None:

        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist,
                        (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = ValueNet(self.edge_index)
        self.action_net = PolicyNet(self.edge_index)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        # Preprocess the observation if needed

        latent_pi = latent_vf = self.mlp_extractor(obs)
        # latent_pi = self.policy_net(latent_pi)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        return self.action_dist.proba_distribution(mean_actions, self.log_std)


    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

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
       # features = self.extract_features(obs)
        latent_pi = latent_vf = self.mlp_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict(self, observation, state, mask, deterministic):
        features = self.features_extractor(observation)
        action = self.action_net(features)
        action = action.detach().numpy().flatten()

        return action


