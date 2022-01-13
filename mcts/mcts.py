import torch
import env_tensor_for_real


class MCTS:
    def __init__(self, value_net, policy_net, config):
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config

    def search(self, init_obs):
        action_history = []
        obs, adj, vel, length_routes = init_obs
        dones, obs, pat2s, pas2s, validities, rewards = env_tensor_for_real.init_step(obs)
        actions = self.choose_actions(obs, pat2s, pas2s)

        for _ in range(47):
            dones, obs, pat2s, pas2s, validities, rewards = env_tensor_for_real.step(actions, obs, adj, vel, length_routes)
            values = self.get_values(obs)
            obs = self.select(obs, rewards, values, validities, dones)
            actions, obs = self.choose_actions(obs, pat2s, pas2s)

        best_action_history =
        return best_action_history

    def select(self, obs, rewards, values, validities, dones):
        obs = obs[torch.logical_not(dones)][validities]
        rewards = rewards[torch.logical_not(dones)][validities]
        values = values[torch.logical_not(dones)][validities]
        values_from_step0 = rewards+values
        best_idx = torch.sort(values_from_step0).indices
        return obs[best_idx[:256]]

    def get_values(self, obs):
        with torch.no_grad():
            values = self.value_net(obs)

        return values

    def choose_actions(self, obs, pat2s, pas2s):
        n_obs = obs.shape[0]
        n_choices = min(pas2s.shape[0], 1024//n_obs)
        with torch.no_grad():
            ps = self.policy_net(obs, pas2s)
        best_idx = torch.sort(ps).indices[:n_choices]

        return pat2s[best_idx], obs[best_idx]

    def get_best_action_history(self):


