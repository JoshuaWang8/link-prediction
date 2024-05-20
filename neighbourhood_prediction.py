from math import log


def jaccard_similarity(graph, test_set):
    """
    Computes Jaccard Similarity between number of shared nodes.

    Parameters:
        graph: Existing graph acting as the training set.
        test_set: List of links to predict on.

    Returns:
        similarity_scores: List of Jaccard similarity scores for each link in the test set.
    """
    similarity_scores = []

    for node_a, node_b in test_set:
        a_neighbours = set([n for n in graph[node_a]])
        b_neighbours = set([n for n in graph[node_b]])

        shared_neighbours = list(a_neighbours.intersection(b_neighbours))

        similarity_scores.append(len(shared_neighbours) / (len(a_neighbours) + len(b_neighbours)))

    return similarity_scores


def adamic_adar_index(graph, test_set):
    """
    Computes the Adamic-Adar index between all nodes in the test set. Quantifies the similarity
    between nodes based on shared neighbours, and returns a list of scores for each link in the
    order of the test set nodes.

    Parameters:
        graph: Existing graph acting as the training set.
        test_set: List of links to predict on.

    Returns:
        index_scores: List of Adamic-Adar index scores for each link in the test set.
    """
    index_scores = []

    for node_a, node_b in test_set:
        a_neighbours = set([n for n in graph[node_a]])
        b_neighbours = set([n for n in graph[node_b]])

        shared_neighbours = list(a_neighbours.intersection(b_neighbours))
        link_index = 0
        
        for node in shared_neighbours:
            degree = graph.degree[node]
            if degree != 0:
                link_index += 1 / (log(degree))

        index_scores.append(link_index)

    return index_scores