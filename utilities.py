import networkx as nx


def load_graph(edge_list):
    """
    Reads and builds a graph from edge list.

    Parameters:
        file: Path to csv file where each line contains two integers separated by a single comma. Each line
        will represent an undirected edge from the node represented by the first integer to the node represented
        by the second integer.

    Returns:
        graph: NetworkX Graph object with nodes and edges from the edge list.
    """
    graph = nx.Graph()
    file = open(edge_list, "r")

    # Add nodes to graph object
    while True:
        line = file.readline()

        if not line:
            break

        start_node, end_node = line.split(",")
        graph.add_edge(int(start_node), int(end_node))

    return graph


def load_test_set_as_list(file):
    """
    Loads the test set as a list of tuples.

    Parameters:
        file: Path to csv where each line contains two integers separated by a single comma.

    Returns:
        test_set: List of links represented as tuples of (start_node, end_node)
    """
    file = open(file, "r")
    test_set = []

    while True:
        line = file.readline()

        if not line:
            break

        start_node, end_node = line.split(",")
        test_set.append((int(start_node), int(end_node)))

    return test_set


def find_top_links(test_set, scores, n=100):
    """
    Finds the top n links from the test set that are most likely to exist in the graph.

    Parameters:
        test_set: List of links to test, where each link is given as a tuple (start_node, end_node).
        scores: Scores for each link in the test set.

    Returns:
        sorted_links: Top n links by scores.
    """
    combined_scores = zip(test_set, scores)
    sorted_links = [link for link, _ in sorted(combined_scores, key=lambda x: x[1], reverse=True)]
    return sorted_links[:n]


def label_top_links(test_set, scores, n=100):
    """
    Labels the top links in the test_set with a 1 if likely to exist and 0 if unlikely to exist.

    Parameters:
        test_set: List of links to test, where each link is given as a tuple (start_node, end_node).
        scores: Scores for each link in the test set.

    Returns:
        labels: List of labels in the same order as the test set
    """
    top_links = find_top_links(test_set, scores, n)
    labels = [1 if link in top_links else 0 for link in test_set]
    return labels


def write_top_links(top_links, file_name="top_links.csv"):
    """
    Writes the top links to a csv file.

    Parameters:
        top_links: List of links to write, where each link is given as a tuple (start_node, end_node).
        file_name: Name of file to write to.
    """
    file = open(file_name, "w")

    for i in range(len(top_links)):
        file.write(str(top_links[i][0]) + "," + str(top_links[i][1]))
        if i != len(top_links) - 1:
            file.write("\n")


def write_full_results(labels, file_name="full_links_results.csv"):
    """
    Writes the labels for each link in the test set in order.

    Parameters:
        labels: List of labels for each link, where 1 indicates the link is predicted to exist and 0
                indicates the link likely does not exist.
        file_name: Name of file to write to.
    """
    file = open(file_name, "w")
    file.write("EdgeID,label\n")
    for i in range(len(labels)):
        file.write(str(i + 1) + "," + str(labels[i]))
        if i != len(labels) - 1:
            file.write("\n")