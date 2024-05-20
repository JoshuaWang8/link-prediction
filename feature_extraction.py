import networkx as nx
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler


def extract_node_features(graph):
    """
    Extracts features for all nodes in a graph. Features that are considered are:
        - Closeness centrality
        - Betweenness centrality
        - PageRank
        - Eigenvector centrality
        - Katz centrality

    Parameters:
        graph: Graph for which all nodes will have their features extracted.

    Returns:
        all_features: Dictionary with node as keys and a list of features as the value.
    """
    closeness_centralities = nx.closeness_centrality(graph)
    betweenness_centralities = nx.betweenness_centrality(graph)
    pageranks = nx.pagerank(graph)
    eigenvector_centralities = nx.eigenvector_centrality(graph)
    katz_centralities = nx.katz_centrality_numpy(graph)

    
    all_features = {}

    for node in sorted(graph.nodes()):
        all_features[node] = [
            closeness_centralities[node],
            betweenness_centralities[node],
            pageranks[node],
            eigenvector_centralities[node],
            katz_centralities[node],
        ]

    # Normalize features
    scaler = StandardScaler()
    feature_matrix = np.array(list(all_features.values()))
    scaled_features = scaler.fit_transform(feature_matrix)
    
    for i, node in enumerate(sorted(graph.nodes())):
        all_features[node] = scaled_features[i]
    
    return all_features


def create_labelled_pairs(graph):
    """
    Creates a dataset of node pairs and corresponding labels for the given graph. Label will be 1 where the two
    nodes are linked in the graph, and 0 where the nodes are not linked.

    Parameters:
        graph: Graph for which the dataset will be created for.

    Returns:
        pair_labels: Pandas DataFrame with columns "node1", "node2" and "label". "node1" and "node2" will be the two nodes,
                     and "label" will be 1 if there is a link between the two nodes, otherwise 0.
    """
    random.seed(8) # For reproducibility of random sampling

    graph_nodes = graph.nodes()
    pair_labels = pd.DataFrame(columns=["node1", "node2", "label"])

    for node_a in graph_nodes:
        neighbours = [n for n in graph[node_a]]
        # Add all neighbours into the dataset
        for node_b in neighbours:
            pair_labels.loc[len(pair_labels)] = [node_a, node_b, 1]

        # Add the same number of random points who are not neighbours
        non_neighbours = [n for n in graph_nodes if n not in neighbours + [node_a]]
        sampled_non_neighbours = random.sample(non_neighbours, len(neighbours))
        for random_node in sampled_non_neighbours:
            pair_labels.loc[len(pair_labels)] = [node_a, random_node, 0]

    return pair_labels


def get_adjacency_matrix(graph):
    """
    Creates adjacency matrix.

    Parameters:
        graph: Graph for which matrix will be created.

    Returns:
        A: Adjacency matrix.
    """
    graph_nodes = sorted(graph.nodes())
    A = np.zeros((len(graph.nodes()), len(graph.nodes())))

    # Populate the adjacency matrix
    for i in graph_nodes:
        neighbor_list = list(graph.neighbors(i))
        for j in graph_nodes:
            if j in neighbor_list:
                A[i][j] = 1

    return A