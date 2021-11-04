import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_nx_graph (routes):
    graph = {}
    for route in routes:
        graph[str(route.station1)] = str(route.station2)

    print(graph)
    nx_graph = nx.Graph()
    for node_id in graph.keys():
        print(node_id)
        zoom = 0.5
        nx_graph.add_node(str(node_id), zoom=zoom)
    for source, connections in graph.items():
        for target in connections:
            nx_graph.add_edge(str(source), str(target), weight=3)

    graph_nx =nx.draw(nx_graph, with_labels=True)
    return graph_nx
   
