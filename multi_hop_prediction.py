import networkx as nx
import numpy as np


def katz_measure(graph, test_set):
    """
    Predicts links using Katz measure.

    Parameters:
        graph: Existing graph acting as the training set.
        test_set: List of links to predict on.

    Returns:
        katz_scores: List of Katz measure scores for each link in the test set. 
    """
    A = nx.to_numpy_array(graph) # Adjacency matrix
    A2 = np.dot(A, A)  # A^2
    A3 = np.dot(A2, A)  # A^3

    # Katz index for paths of length < 4
    katz_matrix = 0.1 * A + (0.1**2) * A2 + (0.1**3) * A3

    katz_scores = []
    for i, j in test_set:
        katz_scores.append(katz_matrix[i, j])

    return katz_scores