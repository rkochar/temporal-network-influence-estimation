import numpy as np
import networkx as nx

def new_infections(graph, train_dataset, beta):
    infections = []
    timestamp, infected = train_dataset[0][2], 0

    for edge in train_dataset:
        # if graph.nodes[edge[0]]['is_infected']
        if np.random.random() < beta and not graph.nodes[edge[1]]['is_infected']:
            if timestamp != edge[2]:
                infections.append((timestamp, infected))
                timestamp = edge[2]
                infected = 1
            else:
                infected += 1

            graph.nodes[edge[1]]['is_infected'] = True
            graph.nodes[edge[1]]['infected_at'] = edge[2]

    if infected != 1:
        infections.append((timestamp, infected))

    cleanup_graph(graph)
    return np.array(infections), graph

def cleanup_graph(graph):
    nx.set_node_attributes(graph, name='infected_at', values=None)
    nx.set_node_attributes(graph, name='is_infected', values=False)
