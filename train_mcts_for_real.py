import random

import networkx as nx
import numpy as np
import torch
from networkx import fast_gnp_random_graph
from tqdm import tqdm

from mcts.trainer import Trainer
from mcts.value_net import ValueNet
from mcts.policy_net import PolicyNet
from utils import ConfigParams


def main():
    config = ConfigParams()
    value_net = ValueNet(config=config)
    policy_net = PolicyNet(config=config)
    trainer = Trainer(value_net=value_net, policy_net=policy_net, config=config)

    for i in range(1024):
        train_mcts(trainer)
        if i % 48 == 0:
            evaluate()


def train_mcts(trainer):
    pi_examples = []
    v_examples = []
    for _ in range(24):
        print("._:")
        obs = generate_random_env()
        pi_exmp, v_exmp = trainer.execute_episodes(init_obs=obs)
        pi_examples.append(pi_exmp)
        v_examples.append(v_examples)
    trainer.optimize(pi_examples, v_examples)


def evaluate():
    pass


def generate_random_env():
    n_max_stations = 1000
    n_stations = int(max(7, n_max_stations * random.random()))
    max_length = 15
    n_max_trains = 50
    n_trains = int(max(7, n_max_trains * random.random()))
    max_speed = 10
    n_max_passenger = 1000
    n_passenger = int(max(7, n_max_passenger * random.random()))
    max_init_delay = 2000
    max_capa_station = 10
    max_capa_route = 5
    max_capa_train = 30

    p = 2 / (n_stations + 1) * 3
    graph = fast_gnp_random_graph(n_stations, p)
    c = nx.k_edge_augmentation(graph, 1)
    graph.add_edges_from(c)
    adj = torch.from_numpy(nx.to_numpy_matrix(graph)).long()
    for e in graph.edges():
        nx.set_edge_attributes(graph, {(e[0], e[1]): {"weight": random.random()*-max_length}})

    d = nx.spring_layout(graph, dim=4, threshold=0.01)
    vectors = np.array([v for v in d.values()])
    vectors = torch.Tensor(vectors)
    length_routes = (-1.*torch.from_numpy(nx.to_numpy_matrix(graph))).float()

    vel = torch.Tensor([max_speed*random.random() for _ in range(n_trains)])
    train_pos_routes = torch.zeros((1, n_trains, n_stations, n_stations))
    for i in range(n_trains):
        j = random.randint(0, n_stations-1)
        k = random.randint(0, n_stations-1)
        train_pos_routes[0, i, j, k] = 1
    train_pos_routes[train_pos_routes==0] = float("NaN")

    train_pos_stations = torch.zeros((1, n_trains, n_stations, n_stations))
    train_pos_stations[...] = float("NaN")

    delay_passenger = torch.zeros((1, n_passenger, (n_stations+n_trains), n_stations))
    for i in range(n_passenger):
        j = random.randint(0, n_stations-1)
        k = random.randint(0, n_stations-1)
        delay_passenger[0, i, j, k] = -random.randint(10, max_init_delay)

    delay_passenger[delay_passenger==0] = float("nan")

    capa_station = torch.Tensor([random.randint(2, max_capa_station) for _ in range(n_stations)])
    capa_route = torch.zeros_like(length_routes)
    capa_route[length_routes!=float("nan")] = max_capa_route
    capa_train = torch.Tensor([random.randint(5, max_capa_train) for _ in range(n_trains)])
    obs = adj, capa_station, capa_route, capa_train, train_pos_stations, train_pos_routes, delay_passenger, length_routes, vel, vectors

    return obs


if __name__ == '__main__':
    main()

