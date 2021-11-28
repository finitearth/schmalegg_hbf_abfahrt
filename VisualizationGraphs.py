import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_nx_graph(station1s, station2s):
    graph = list(zip(station1s, station2s))

    nx_graph = nx.Graph()
    for station in set(station1s):
        nx_graph.add_node(station)
    for source, target in graph:
        nx_graph.add_edge(source, target)

    # graph.sort(key=lambda t: (int(t[0].__repr__()), int(t[1].__repr__())))
    # print(graph)
    # print(len(station1s)**0.5)

    graph_nx = nx.draw(nx_graph, with_labels=True)
    return graph_nx
   
