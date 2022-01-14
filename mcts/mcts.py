import torch
import env_tensor_for_real
from torch.nn.utils.rnn import pad_sequence

batch_size = 256


class MCTS:
    def __init__(self, value_net, policy_net, config):
        self.value_net = value_net
        self.policy_net = policy_net
        self.config = config
        self.action_histories_dones = None
        self.action_histories = None
        self.rewards = None
        self.rewards_dones = None

    def search(self, init_obs):
        dones, obs, pat2s, pas2s, validities, rewards = env_tensor_for_real.init_step(init_obs)
        obs, actions = self.choose_actions(obs, pat2s, pas2s)

        for _ in range(47):
            dones, obs, pat2s, pas2s, validities, rewards = env_tensor_for_real.step(actions, obs)
            values = self.get_values(obs)
            obs = self.select(obs, rewards, values, validities, dones)
            actions, obs = self.choose_actions(actions, obs, pat2s, pas2s)
            self.update_action_history(actions, validities, dones, rewards)

        return self.get_best_action_history()

    def select(self, obs, rewards, values, validities, dones):
        obs = obs[torch.logical_not(dones)][validities]
        rewards = rewards[torch.logical_not(dones)][validities]
        values = values[torch.logical_not(dones)][validities]
        values_from_step0 = rewards + values
        best_idx = torch.sort(values_from_step0).indices

        return obs[best_idx[:batch_size // 4]]

    def get_values(self, obs):
        with torch.no_grad():
            values = self.value_net(obs)

        return values

    def choose_actions(self, obs, pat2s, pas2s):
        if pat2s is None: return obs, None
        n_obs = obs[-1].shape[0]
        n_choices = min(pas2s.shape[0], batch_size // n_obs)
        with torch.no_grad():
            ps = self.policy_net(obs, pas2s)
        best_idx = torch.sort(ps).indices[:n_choices]

        return pat2s[best_idx], obs[best_idx]

    def update_action_history(self, actions, validities, dones, rewards):
        actions_not_done = actions[torch.logical_not(dones)][validities]
        actions_done = actions[dones][validities]
        rewards_not_done = rewards[torch.logical_not(dones)][validities]
        rewards_done = rewards[dones][validities]

        if self.action_histories is None:
            self.action_histories = actions_not_done
        else:
            action_histories_not_done = self.action_histories[torch.logical_not(dones)]
            self.action_histories = pad_sequence([action_histories_not_done, actions_not_done])

        if self.action_histories_dones is None:
            self.action_histories_dones = actions_done
        else:
            action_histories_done = self.action_histories[dones]
            self.action_histories_dones = pad_sequence([action_histories_done, actions_done])

        if self.rewards is None:
            self.rewards = rewards_not_done
        else:
            rewards_not_done_history = self.action_histories[torch.logical_not(dones)]
            self.rewards = pad_sequence([rewards_not_done_history, rewards_not_done])

        if self.rewards_dones is None:
            self.rewards_dones = rewards_done
        else:
            rewards_done_history = self.action_histories[dones]
            self.rewards = pad_sequence([rewards_done_history, rewards_not_done])

    def get_best_action_history(self):
        dones_action_histories = self.action_histories_dones
        return max(dones_action_histories, key=lambda x: x[1])[0]
