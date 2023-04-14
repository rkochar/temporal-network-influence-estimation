import networkx as nx
import numpy.random
import pandas as pd


def compute_influences(
    graph: nx.MultiGraph,
    dataset: pd.DataFrame,
    delta_t: int,
    beta: float,
    rng: numpy.random.Generator = numpy.random.default_rng(),
):
    node_influences = {}

    for node in graph.nodes():
        # Add infection time as a node attribute and set it to infinity for all nodes.
        nx.set_node_attributes(graph, float("inf"), name="infected_at")

        # Set the seed node to be infected at t=-1.
        # The time we set here doesn't matter as long as it's t0 or lower.
        graph.nodes[node]["infected_at"] = -1

        # Find the first contact of the node and use that as t0.
        # NOTE: Doing this doesn't make sense to me; if we pick a new t0 for every node,
        # we're computing the influence of a different sub-network for every node.
        # If t0 and t0 + delta_t is different for every node, we're ranking the nodes
        # based on how influential they are within in their own sub-network, whereas
        # I think we should be a ranking for a specific [t0, t0 + delta_t].
        # We can't compare the influences if we're constantly moving the goalposts, no?
        first_source = dataset[dataset["source"] == node].head(1)["ts"].to_numpy()
        first_target = dataset[dataset["target"] == node].head(1)["ts"].to_numpy()

        if len(first_source) == 0:
            first_source = [float("inf")]

        if len(first_target) == 0:
            first_target = [float("inf")]

        t0 = min(first_source[0], first_target[0])

        # For every timestamp in [t0, t0 + delta_t],
        # go through the neighbors of each node and infect them if
        # the source node has been infected at an earlier timestamp.
        # NOTE: I don't think we should be using phi here. We want to compare our
        # results to the *actual* influence ranking of the nodes, and we don't know
        # the actual influence if we don't consider the entirety of
        # [t0, t0 + delta_t].
        range_end = t0 + delta_t + 1
        for t in range(t0, range_end):
            for n, nbrs in graph.adjacency():
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

        node_influences[node] = len(
            [n for n in graph.nodes if graph.nodes[n]["infected_at"] <= float("inf")]
        ) / len(graph.nodes)
    return node_influences


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
