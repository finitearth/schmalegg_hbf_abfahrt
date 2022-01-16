import torch
import env_tensor_for_real
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils.sparse import dense_to_sparse
from tqdm import tqdm

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
        actions, obs = self.choose_actions(obs, pat2s, pas2s)

        for _ in tqdm(range(47)):
            dones, obs, pat2s, pas2s, validities, rewards = env_tensor_for_real.step(actions, obs)
            values = self.get_values(obs)
            obs = self.select(obs, rewards, values, validities, dones)
            actions, obs = self.choose_actions(obs, pat2s, pas2s)
            self.update_action_history(actions, validities, dones, rewards)

        return self.get_best_action_history()

    def select(self, obs, rewards, values, validities, dones):
        # obs = obs[validities] #[torch.logical_not(dones)]
        rewards = rewards[validities] #[torch.logical_not(dones)]
        values = values[validities] #[torch.logical_not(dones)]
        values_from_step0 = rewards + values
        best_idx = torch.sort(values_from_step0).indices

        return obs#[best_idx[:batch_size // 4]]

    def get_values(self, obs):
        adj, _, _, _, _, _, delay_passenger, _, train_pos_routes, _, vectors = obs

        pass_adj = torch.where(delay_passenger.isnan(), torch.tensor(0, dtype=torch.float32), delay_passenger)
        n_stationsplustrains = vectors.shape[1]
        pass_adj = pass_adj.sum(-3)
        pass_adj = pass_adj[:, :n_stationsplustrains, :]
        vectors = vectors#[:, :n_stationsplustrains, :]

        train_adj = torch.where(train_pos_routes.isnan(), torch.tensor(0, dtype=torch.float32), train_pos_routes)
        train_adj = train_adj.sum(dim=1)
        train_adj = train_adj[:, :n_stationsplustrains, :n_stationsplustrains]
        batch = torch.div(vectors[0, :, 0].nonzero(as_tuple=True)[0], n_stationsplustrains, rounding_mode='floor')
        vectors = vectors.squeeze()
        (adj, adj_attr), (pass_adj, pass_adj_attr),  (train_adj, train_adj_attr) = dense_to_sparse(adj), dense_to_sparse(pass_adj), dense_to_sparse(train_adj)
        with torch.no_grad():
            values = self.value_net(vectors, adj, adj_attr, pass_adj, pass_adj_attr,  train_adj, train_adj_attr, batch)

        return values

    def choose_actions(self, obs, pat2s, pas2s):

        if pat2s is None or pat2s.shape[0] == 0: return None, obs

        n_acs = pat2s.shape[0]
        print(n_acs)
        n_obs = obs[-3].shape[-1]
        n_choices = min(pas2s.shape[0], batch_size // n_obs)
        adj, _, _, _, _, _, delay_passenger, _, train_pos_routes, _, vectors = obs
        if batch_size<n_acs:
            pass_adj = torch.where(delay_passenger.isnan(), torch.tensor(0, dtype=torch.float32), delay_passenger)
            n_stationsplustrains = vectors.shape[1]
            pass_adj = pass_adj.sum(-3)
            pass_adj = pass_adj[:, :n_stationsplustrains, :]

            train_adj = torch.where(train_pos_routes.isnan(), torch.tensor(0, dtype=torch.float32), train_pos_routes)
            train_adj = train_adj.sum(dim=1)
            train_adj = train_adj[:, :n_stationsplustrains, :n_stationsplustrains]
            batch = torch.div(vectors[0, :, 0].nonzero(as_tuple=True)[0], n_stationsplustrains, rounding_mode='floor')
            vectors = vectors.squeeze()
            (adj, adj_attr), (pass_adj, pass_adj_attr), (train_adj, train_adj_attr) = dense_to_sparse(adj), dense_to_sparse(
                pass_adj), dense_to_sparse(train_adj)

            with torch.no_grad():
                ps = self.policy_net(pas2s, vectors, adj, adj_attr, pass_adj, pass_adj_attr, train_adj, train_adj_attr, batch)
            best_idx = torch.sort(ps).indices[..., :n_choices]
            adj, capa_station, capa_route, capa_train, train_pos_stations, train_progress, delay_passenger, length_routes, train_pos_routes, vel, vectors = obs
            obs = adj, capa_station, capa_route, capa_train, train_pos_stations.expand(n_choices, -1, -1, -1), train_progress.expand(n_choices, -1, -1, -1), delay_passenger.expand(n_choices, -1, -1, -1), length_routes, train_pos_routes.expand(n_choices, -1, -1, -1), vel, vectors
            adj, capa_station, capa_route, capa_train, train_pos_stations, train_progress, delay_passenger, length_routes, train_pos_routes, vel, vectors = obs
            obs = adj, capa_station, capa_route, capa_train, train_pos_stations[best_idx, ...][0], train_progress[best_idx, ...][0], delay_passenger[best_idx, ...][0], length_routes, train_pos_routes[best_idx, ...][0], vel, vectors
            return pat2s, obs

        else:
            n_m = max(1, n_acs)
            adj, capa_station, capa_route, capa_train, train_pos_stations, train_progress, delay_passenger, length_routes, train_pos_routes, vel, vectors = obs
            obs = adj, capa_station, capa_route, capa_train, train_pos_stations.expand(n_m, -1, -1, -1), train_progress.expand(n_m, -1, -1, -1), delay_passenger.expand(n_m, -1, -1, -1), length_routes, train_pos_routes.expand(n_m, -1, -1, -1), vel, vectors
            return pat2s, obs

    def update_action_history(self, actions, validities, dones, rewards):
        if actions is None: return
        actions_not_done = actions[validities] # [torch.logical_not(dones)]
        actions_done = actions[validities] # [dones]
        rewards_not_done = rewards[validities] #[torch.logical_not(dones)]
        rewards_done = rewards[validities] #[dones]

        if self.action_histories is None:
            self.action_histories = [actions_not_done]
        else:
            # action_histories_not_done = self.action_histories#[torch.logical_not(dones)]
            self.action_histories += [actions_not_done]#pad_sequence([action_histories_not_done, actions_not_done], batch_first=False)

        if self.action_histories_dones is None:
            self.action_histories_dones = [actions_done]
        else:
            # action_histories_done = self.action_histories#[dones]
            self.action_histories_dones += [actions_done]# pad_sequence([action_histories_done, actions_done])

        if self.rewards is None:
            self.rewards = [rewards_not_done]
        else:
            # rewards_not_done_history = self.action_histories#[torch.logical_not(dones)]
            self.rewards +=[rewards_not_done]# pad_sequence([rewards_not_done_history, rewards_not_done])

        if self.rewards_dones is None:
            self.rewards_dones = [rewards_done]
        else:
            # rewards_done_history = self.action_histories#[dones]
            self.rewards += [rewards_not_done]#pad_sequence([rewards_done_history, rewards_not_done])

    def get_best_action_history(self):
        dones_action_histories = self.action_histories_dones
        return None, None#max(dones_action_histories, key=lambda x: x[1])[0]


class StepLogger:
    def __init__(self, passenger_str, trains_str, stationandtrains_str, route_str, routes_dict):
        self.passenger_str = passenger_str
        self.trains_str = trains_str
        self.route_str = route_str
        self.stationsandtrains_str = stationandtrains_str
        self.passenger_log = {p: [] for p in passenger_str}
        self.train_log = {t: [] for t in trains_str}
        self.routes_dict = routes_dict
        self.step = 1

    def inc_step(self):
        self.step += 1

    def boarding_passenger(self, tensor):
        for stationtrain, passenger in tensor:
            stationtrain = self.stationsandtrains_str[stationtrain]
            self.passenger_log[self.passenger_str[passenger]].append(f"{self.step} Board {stationtrain}")

    def redirect_train(self, s2s, t2s):
        for (train, station2), (station1, _) in zip(t2s.squeeze(), s2s.squeeze()):
            direct_to = self.routes_dict[(station1, station2)]
            self.train_log[self.trains_str[train]].append(f"{self.step} Depart {direct_to}")

    def init_train(self, train_idx, start_station):
        start_station = self.stationsandtrains_str[start_station]
        self.train_log[self.trains_str[train_idx]].append(f"0 Start {start_station}")

    def save_log(self, filename):
        log = ""
        for p, c in self.passenger_log.values():
            log += f"\n[Passenger:{p}\n]"
            for s in c:
                log += s + "\n"

        for t, c in self.train_log.values():
            log += f"\n[Train:{t}]\n"
            for s in c:
                log += s + "\n"

        with open(filename, "w") as f:
            f.write(log)
