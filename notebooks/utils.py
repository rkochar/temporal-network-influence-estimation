import networkx as nx
import numpy as np
import numpy.random
import pandas as pd


def compute_influence(
    node: int,
    graph: nx.MultiGraph,
    t0: int,
    delta_t: int,
    beta: float,
    rng: numpy.random.Generator = numpy.random.default_rng(),
):

    # Add infection time as a node attribute and set it to infinity for all nodes.
    nx.set_node_attributes(graph, float("inf"), name="infected_at")

    # Set the seed node to be infected at t=-1.
    # The time we set here doesn't matter as long as it's t0 or lower.
    graph.nodes[node]["infected_at"] = -1

    # For every timestamp in [t0, t0 + delta_t],
    # go through the neighbors of each node and infect them if
    # the source node has been infected at an earlier timestamp.
    range_end = t0 + delta_t + 1
    # NOTE switching the order of these two for loops influences performance
    for n, nbrs in graph.adjacency():
        for t in range(t0, range_end):
            if graph.nodes[n]["infected_at"] < t:
                # The source node is infected,
                # check neighbors and see if they meet at the current timestamp.
                for neighbor, contacts in nbrs.items():
                    if graph.nodes[neighbor]["infected_at"] <= t:
                        # The neighbor has already been infected; we want to
                        # preserve the original infection time, so let's skip to the
                        # next neighbor.
                        continue
                    if t in contacts.keys():
                        # The nodes meet at this timestamp;
                        # use beta to decide if we should infect.
                        if rng.uniform() <= beta:
                            graph.nodes[neighbor]["infected_at"] = t

    return len(
        [n for n in graph.nodes if graph.nodes[n]["infected_at"] < float("inf")]
    ) / len(graph.nodes)


def compute_influences(
    graph: nx.MultiGraph,
    t0: int,
    delta_t: int,
    beta: float,
    rng: numpy.random.Generator = numpy.random.default_rng(),
):
    node_influences = {}

    for node in graph.nodes():
        node_influences[node] = compute_influence(node, graph, t0, delta_t, beta, rng)

    return node_influences


def compute_avg_influences(
    graph: nx.MultiGraph,
    t0: int,
    delta_t: int,
    beta: float,
    iters: int,
    rng: numpy.random.Generator = numpy.random.default_rng(),
):
    # Iterate iter times and compute all node influences on each time
    node_influences_aggr = []
    for i in range(iters):
        node_influences_dict = compute_influences(graph, t0, delta_t, beta, rng)

        node_influences = np.zeros(len(graph.nodes))
        for k, v in node_influences_dict.items():
            # -1 for nx's 1-indexing
            node_influences[k - 1] = v

        node_influences_aggr.append(node_influences)

    # Convert list to np array
    node_influences_aggr = np.array(node_influences_aggr)
    # Compute the mean
    node_avg_influences = node_influences_aggr.mean(axis=0)
    # print(node_influences_aggr.std(axis=0))

    # Compute the argsort (indices that sort the array) as the ranking order
    node_ranking_asc = node_influences.argsort()
    node_ranking_asc = node_ranking_asc + 1  # translate back to the nx's one-indexing

    return node_avg_influences, node_ranking_asc


def slice_dataset(data: pd.DataFrame, t0: int, delta_t: int, phi: float):
    # Calculate the cutoff point.
    cutoff = t0 + delta_t * phi
    # Make a copy of the data.
    data_slice = data.copy()
    # Slice, starting at t0.
    data_slice = data_slice[data_slice["ts"] >= t0]
    # Slice again, but this time remove anything after the cutoff.
    data_slice = data_slice[data_slice["ts"] <= cutoff]
    return data_slice
