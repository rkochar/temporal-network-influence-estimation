import yaml
import numpy as np

import src.predict as predict_functions
import src.metric as metric_functions
import src.weight as weight_functions

import networkx as nx
# import matplotlib.pyplot as plt
from collections import Counter

def get_config():
    """
    Read config.yaml file.
    :return:
    """
    # TODO: Better error handling
    with open("./../config.yaml", "r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)


def parse_config():
    """
    Parse config.yaml file.
    :return: configurations.
    """
    config = get_config()
    datasets = config['datasets']
    beta = config['beta']
    total_length = config['length']
    phi = config['phi']

    weight_function = weight_functions.get_weight_function(config['weight_function'])
    metric_function = metric_functions.get_metric_function(config['metric_function'])
    predict_function = predict_functions.get_predict_function(config['prediction_function'])

    return datasets, beta, total_length, phi, weight_function, metric_function, predict_function


def get_dataset(dataset, total_length, phi):
    """
    Load dataset and do some basic stuff.

    :param dataset: name of txt file
    :param total_length: to be studied
    :param phi: the split
    :return: lots of things.
    """
    data = np.loadtxt("./../data/" + dataset + ".txt", delimiter="\t", dtype=np.int32)

    len_dataset = len(data)
    length_to_use = int(total_length * len_dataset / 100)
    start = np.random.randint(0, len_dataset - length_to_use)
    train_length = int(length_to_use * phi / 100)
    predict_length = length_to_use - train_length

    # Find influence
    graph = make_graph(dataset=dataset)
    influences = find_influence(graph=graph, edges=data[start : start + length_to_use], beta=beta)
    graph = influence_of_each_node(graph=graph, train_dataset=data[start : start + train_length], beta=beta)

# Extract parts of dataset we need.
    train_dataset = data[start:start + train_length]
    test_dataset = data[start + train_length:start + length_to_use]

    return influences, graph, train_dataset, test_dataset


def influence_of_each_node(graph, train_dataset, beta):
    """
    Find influence of each node.

    :param graph:
    :param train_dataset:
    :param beta:
    :return:
    """
    graph.nodes[train_dataset[0][0]]["is_infected"] = True
    graph.nodes[train_dataset[0][0]]["infected_at"] = train_dataset[0][2]

    for i, edge in enumerate(train_dataset):
        if graph.nodes[edge[0]]["influence"] == -1:
            find_influence_of_node(train_dataset[i:], edge[0], graph)

    return graph


def find_influence_of_node(edges, node, original_graph):
    graph = make_graph(dataset=dataset)
    graph.nodes[node]["is_infected"] = True
    graph.nodes[node]["infected_at"] = edges[0][2]
    infected = 0

    for edge in edges:
        if not graph.nodes[edge[1]]['is_infected']:
            infected += 1
            graph.nodes[edge[1]]['is_infected'] = True
            graph.nodes[edge[1]]['infected_at'] = edge[2]
    original_graph.nodes[node]["influence"] = infected


def make_graph(dataset):
    """
    Make graph. Currently works as a dict for nodes. Sort of like a custom Node class.
    :param dataset:
    :return:
    """
    edges = np.loadtxt("./../data/" + dataset + ".txt", delimiter="\t", dtype=np.int32)
    nodes = np.union1d(np.unique(edges[:, 0]), np.unique(edges[:, 1]))
    graph = nx.Graph()

    for node in nodes:
        graph.add_node(node, is_infected=False, infected_at=None, influence=-1)

    # for edge in dataset:
    #     graph.add_edge(edge[0], edge[1], edge[3])

    # for row in train_dataset:
    #     graph.add_node(row[0], is_infected=False, infected_at=None)
    #     graph.add_node(row[1], is_infected=False, infected_at=None)
    return graph


def cleanup_graph(graph):
    """
    Clean up graph so it can be reused.
    :param graph:
    :return:
    """
    nx.set_node_attributes(graph, name='infected_at', values=None)
    nx.set_node_attributes(graph, name='is_infected', values=False)


def find_influence(graph, edges, beta):
    """
    Calculate influence at each timestep.

    :param graph:
    :param edges:
    :param beta: chance to get infected
    :return: list of tuples [(timestamp, influence)]
    """
    influence, timestamp, infected = [], edges[0][2], 0
    # cnt = Counter(edges[:, 2])
    for edge in edges:
        if np.random.random() <= beta and not graph.nodes[edge[1]]['is_infected']:
            if timestamp != edge[2]:
                influence.append((timestamp, infected))
                timestamp = edge[2]
                infected = 1
            else:
                infected += 1
        graph.nodes[edge[1]]['is_infected'] = True
        graph.nodes[edge[1]]['infected_at'] = edge[2]

    if infected != 1:
        influence.append((timestamp, infected))

    cleanup_graph(graph)
    x = np.array(influence)
    x[:, 1] = np.cumsum(x[:, 1])
    return x


def run_experiment(graph, beta, train_dataset, num_predict, weight_function, metric_function, predict_function):
    """
    Run an experiment.

    :param graph:
    :param beta:
    :param train_dataset:
    :param num_predict:
    :param weight_function:
    :param metric_function:
    :param predict_function:
    :return:
    """
    # Use metric function
    infections, graph = metric_function(graph=graph, train_dataset=train_dataset, beta=beta)

    # Apply weight function
    infections[:, 1] = weight_function(data=infections[:, 1])

    # Make prediction
    predicted = predict_function(data=infections, new_points=num_predict)

    print(predicted)

    return predicted


# def compare(predicted, actual):
#     """
#     Compare predicted and actual values.
#
#     :param predicted:
#     :param actual:
#     :return:
#     """
#     plt.plot(predicted, color='red')
#     plt.plot(actual[:, 1], color='blue')
#     plt.show()


if __name__ == '__main__':
    """
    Main function.
    """
    datasets, beta, total_length, phi, weight_function, metric_function, predict_function = parse_config()

    for dataset in datasets:
        influences, graph, train_dataset, predict_dataset = get_dataset(dataset=dataset, total_length=total_length, phi=phi)
        predicted = run_experiment(graph=graph,
                                   beta=beta,
                                   train_dataset=train_dataset,
                                   num_predict=len(influences),
                                   weight_function=weight_function,
                                   metric_function=metric_function,
                                   predict_function=predict_function)

        # Compare prediction with actual value
        # compare(predicted=predicted, actual=influences)
