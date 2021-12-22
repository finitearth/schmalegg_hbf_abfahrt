import json
import random

from matplotlib import pyplot as plt
from networkx.generators.random_graphs import *
import networkx as nx
import objects
from networkx.readwrite.adjlist import *


n = 100
for i in range(1, 5):#[5, 20, 30, 50, 100]
    nn = int(max(7, n))
    p = 2/(nn+1) # n*(n+1)/2 = n + n - 1 +... + 1
    graph = fast_gnp_random_graph(nn, p)
    c = nx.k_edge_augmentation(graph, 1)
    graph.add_edges_from(c)
    nx.draw(graph)
    plt.show()

# import glob
#
# for file in glob.glob("./graphs/eval/*"):
#     text = {}
#     graph = read_adjlist(file)
#     text["routes"] = list(graph.edges)
#     stations = []
#     for route in graph.edges:
#         for station in route:
#             stations.append({
#                 "name": station,
#                 "capacity": random.randint(1, 5)
#             })
#
#     text["stations"] = stations
#
#     passengers = []
#     n_passengers = random.randint(1, 20)
#     for _ in range(n_passengers):
#         passengers.append(
#             {"n_people": random.randint(1, 20), "target_time": random.randint(5, 100)}
#         )
#     text["passengers"] = passengers
#
#     trains = []
#     n_trains = random.randint(1, 10)
#     for _ in range(n_trains):
#         trains.append({"capacity": random.randint(5, 100), "station": int(random.random()*len(stations))})
#     text["trains"] = trains
#     name = file.split(".txt")[0]
#     with open(name+".json", 'w') as f:
#         json.dump(text, f)


# import glob
# import objects
#
# for i, file in enumerate(glob.glob("./graphs/eval/*")):
#     a = objects.EnvBlueprint()
#     a.read(file)
#     break
#

