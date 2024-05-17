from utilities import *
from neighbourhood_prediction import adamic_adar_index


if __name__ == "__main__":
    training_file = "./data/trainingset.csv"
    test_file = "./data/testset.csv"

    train_graph = load_graph(training_file)
    test_list = load_test_set_as_list(test_file)

    # Adamic-Adar Index Scoring
    aa_index = adamic_adar_index(train_graph, test_list)
    write_top_links(find_top_links(test_list, aa_index))
    write_full_results(label_top_links(test_list, aa_index))