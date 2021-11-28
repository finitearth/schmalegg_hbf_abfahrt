import generate_envs
import VisualizationGraphs

import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    for _ in range(1):
        routes = generate_envs.generate_random_routes()[0]
    VisualizationGraphs.create_nx_graph(routes[0], routes[1])
    plt.show()
