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