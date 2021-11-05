import generate_random
import VisualizationGraphs

import matplotlib.pyplot as plt

if __name__ == '__main__':
    routes = generate_random.generate_random_routes()

    for route in routes:
        print(route)
    VisualizationGraphs.create_nx_graph(routes)
    plt.show()
