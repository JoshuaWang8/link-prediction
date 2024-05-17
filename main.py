from load_data import load_graph


if __name__ == "__main__":
    training_file = "./data/trainingset.csv"

    train_graph = load_graph(training_file)

    print(train_graph.number_of_edges())