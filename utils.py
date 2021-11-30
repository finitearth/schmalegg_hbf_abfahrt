import networkx as nx


def create_nx_graph(station1s, station2s):
    graph = list(zip(station1s, station2s))

    nx_graph = nx.Graph()
    for station in set(station1s):
        nx_graph.add_node(station)
    for source, target in graph:
        nx_graph.add_edge(source, target)

    graph_nx = nx.draw(nx_graph, with_labels=True)
    return graph_nx
