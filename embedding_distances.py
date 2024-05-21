import torch


def embedding_distance_scores(node_embeddings, test_list):
    """
    Scores potential links from the test set by comparing distances between node embeddings.

    Parameters:
        node_embeddings: Embeddings of all nodes in the graph.
        test_list: List of links represented as (node_a, node_b) tuples to predict on.

    Returns:
        distance_scores: Inverted distances between nodes for each link.
    """
    distances = []

    for i in range(len(test_list)):
        node1_embedding = node_embeddings[test_list[i][0]]
        node2_embedding = node_embeddings[test_list[i][1]]
        embedding_dist = torch.sum((node1_embedding - node2_embedding) ** 2)
        distances.append(embedding_dist.item())

    # Invert scores since a higher score means more likely link
    distance_scores = [-1 * dist for dist in distances]

    return distance_scores