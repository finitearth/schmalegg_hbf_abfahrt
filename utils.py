import cv2
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

import wandb
from PIL import Image, ImageDraw
import networkx as nx
import numpy as np
import torch
from stable_baselines3.common.logger import Logger, make_output_format, KVWriter
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, GraphConv, GCN2Conv, DenseGCNConv, FastRGCNConv, \
    ResGatedGraphConv
import env
from policies import ppo_policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ConfigParams:
    def __init__(self, wandb_config=None):
        self.lr_pi = 10 ** -4
        self.lr_v = 10 ** -4
        self.n_iters = 4
        self.n_eps = 1
        self.n_epochs = 4
        w = wandb_config is not None
        self.vf_coef = wandb_config.vf_coef if w else .5
        self.learning_rate = wandb_config.learning_rate if w else 10 ** -4
        self.gamma = wandb_config.gamma if w else 0.99
        self.clip_range = wandb_config.clip_range if w else .2

        self.n_node_features = wandb_config.n_node_features if w else 4
        self.action_vector_size = wandb_config.action_vector_size if w else 8
        self.hidden_neurons = wandb_config.hidden_neurons if w else 8
        self.it_b4_dest = wandb_config.it_b4_dest if w else 4
        self.it_aft_dest = wandb_config.it_aft_dest if w else 4
        self.normalize = wandb_config.normalize if w else True
        self.log_std_init = wandb_config.log_std_init if w else -2

        self.reward_per_step = wandb_config.reward_per_step if w else 0.  # -.05
        self.reward_reached_dest = wandb_config.reward_reached_dest if w else 0.#1.
        self.reward_step_closer = wandb_config.reward_step_closer if w else .1

        self.aggr_dest = wandb_config.aggr_dest if w else "add"
        self.aggr_con = wandb_config.aggr_con if w else "add"

        self.n_envs = 4
        self.batch_size = wandb_config.batch_size if w else 32
        n_steps = wandb_config.n_steps if w else 32
        self.n_steps = n_steps + self.batch_size - (
                    n_steps % self.batch_size)  # such that batch_size is a factor of n_steps
        self.total_steps = self.n_steps * self.n_envs * self.batch_size

        self.n_lin_extr = wandb_config.n_lin_extr if w else 1
        self.n_lin_policy = wandb_config.n_lin_policy if w else 1
        self.n_lin_value = wandb_config.n_lin_value if w else 2

        self.range_inputvec = wandb_config.range_inputvec if w else .05

        self.env = env
        self.policy = ppo_policy

        activation_str = wandb_config.activation if w else "none"
        activations = {"tanh": torch.tanh, "relu": torch.relu, "none": nn.Identity()}
        self.activation = activations[activation_str]

        conv_str = wandb_config.conv if w else "GraphConv"  #
        convs = {"GCNConv": GCNConv, "SAGEConv": SAGEConv, "GraphConv": GraphConv,
                 "ResGatedGraphConv": ResGatedGraphConv}
        self.conv = convs[conv_str]


class CustomLogger(Logger):
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb
        super(CustomLogger, self).__init__("", [make_output_format("stdout", "./logs", "")])

    def dump(self, step: int = 0) -> None:
        """
        Write all of the diagnostics from the current iteration
        """
        for _format in self.output_formats:
            if isinstance(_format, KVWriter):
                _format.write(self.name_to_value, self.name_to_excluded, step)
        if self.use_wandb: wandb.log(self.name_to_value)
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()


def create_nx_graph(station1s, station2s):
    graph = list(zip(station1s, station2s))
    nx_graph = nx.Graph()
    for station in set(station1s):
        nx_graph.add_node(station)
    for source, target in graph:
        nx_graph.add_edge(source, target)

    graph_nx = nx.draw(nx_graph, with_labels=True)
    return graph_nx


def draw_arrow(im, p0, p1, thickness=1, color=(0, 0, 0)):
    na = np.array(im)
    na = cv2.arrowedLine(na, p0, p1, color, thickness)
    im = Image.fromarray(na)
    return ImageDraw.Draw(im), im


def collate(batch):
    # targets = []
    actions = []
    for elem in batch:
        # targets.append(elem.target)
        actions.append(elem.actions)
    # target = pad_sequence(targets, batch_first=True)
    actions = pad_sequence(actions, batch_first=True, padding_value=-1)#.flatten(start_dim=1)
    actions = actions.squeeze(2)
    batch = Batch.from_data_list(batch, exclude_keys=["action"])
    # batch.target = target
    batch.actions = actions
    return batch



def convert_observation(obs, config):
    datas = []
    if not isinstance(obs, dict):
        if isinstance(obs, np.ndarray):
            if None in obs: raise ValueError("._.")
            obs = torch.from_numpy(obs)
        if len(obs.shape) < 2: obs = obs.unsqueeze(0)
        n_batches = obs.shape[0]

        for i in range(n_batches):
            n_stations = int(obs[i, -1])
            n_edges = int(obs[i, -2])
            n_passenger = int(obs[i, -3])
            n_trains = int(obs[i, -4])

            edge_index_connections0 = obs[i, :n_edges]
            edge_index_connections1 = obs[i, 25_000:25_000 + n_edges]
            edge_index_connections = torch.vstack((edge_index_connections0, edge_index_connections1)).long()

            edge_index_destinations0 = obs[i, 50_000:50_000 + n_passenger]
            edge_index_destinations1 = obs[i, 75_000:75_000 + n_passenger]
            edge_index_destinations = torch.vstack((edge_index_destinations0, edge_index_destinations1)).long()

            edge_index_trains0 = obs[i, 100_000:100_000 + n_trains]
            edge_index_trains1 = obs[i, 125_000:125_000 + n_trains]
            edge_index_trains = torch.vstack((edge_index_trains0, edge_index_trains1)).long()

            input_vectors = obs[i, 150_000:150_000 + (n_stations + n_trains) * config.n_node_features]
            input_vectors = torch.reshape(input_vectors, ((n_stations + n_trains), config.n_node_features)).float()

            data = CustomData(x=input_vectors,
                              edge_index_connections=edge_index_connections,
                              edge_index_destinations=edge_index_destinations,
                              edge_index_trains=edge_index_trains)

            datas.append(data)

    else:
        n_batches = 1
        n_stations = obs["n_stations"]
        n_trains = obs["n_trains"]
        input_vectors = torch.Tensor(obs["input"]).long()
        edge_index_connections = torch.Tensor(obs["eic"]).long()
        edge_index_trains = torch.Tensor(obs["eit"]).long()
        edge_index_destinations = torch.Tensor(obs["eid"]).long()
        input_vectors = torch.reshape(input_vectors, ((n_stations + n_trains), config.n_node_features)).float()

        batch = torch.zeros(len(input_vectors), dtype=torch.int64)

        return input_vectors, edge_index_connections, edge_index_destinations, edge_index_trains, batch

    data_loader = DataLoader(datas, batch_size=n_batches, shuffle=False)
    b = next(iter(data_loader))

    return b.x.to(device), b.edge_index_connections.to(device), b.edge_index_destinations.to(device), b.edge_index_trains.to(device), b.batch.to(device)


class CustomData(Data):
    def __init__(self, x=None, edge_index_connections=None, edge_index_destinations=None, edge_attr=None,
                 edge_index_trains=None, **kwargs):
        super().__init__(x=x, edge_attr=edge_attr, **kwargs)
        if edge_index_connections is not None:
            edge_index_connections, _ = add_self_loops(edge_index_connections)
        self.edge_index_destinations = edge_index_destinations
        self.edge_index_connections = edge_index_connections
        self.edge_index_trains = edge_index_trains


def set_node_attributes(graph, stations, config):
    d = nx.spring_layout(graph, dim=config.n_node_features)
    for s in stations:
        s.set_input_vector(d[int(s)], config=config)


def cart_prod(tensor):
    D = tensor.reshape(-1, 1)
    _B = torch.stack([x.repeat(x.size(0)) for x in tensor]).reshape(-1)
    _D = D.view(-1, 1).expand(tensor.size(0)*tensor.size(1), tensor.size(1)).reshape(-1)
    return torch.stack([_D, _B]).T.reshape(tensor.size(0), tensor.size(1)**2, 2)



def to_sparse_tensor(tensor):
    sparse_copy = tensor.to_sparse()
    sparse_tensor = SparseTensor(row=sparse_copy.indices()[0], col=sparse_copy.indices()[1], value=sparse_copy.values(), sparse_sizes=sparse_copy.size())

    return sparse_tensor

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

