from enviroments.generate_envs import generate_random_env
from utils import create_nx_graph

import matplotlib.pyplot as plt

if __name__ == '__main__':
    routes, _, _ = generate_random_env(max_n_stations=30)
    create_nx_graph(routes[0], routes[1])
    plt.show()
