import torch
from utilities import *
from neighbourhood_prediction import adamic_adar_index
from feature_extraction import *
from gcn import *


if __name__ == "__main__":
    training_file = "./data/trainingset.csv"
    test_file = "./data/testset.csv"

    train_graph = load_graph(training_file)
    test_list = load_test_set_as_list(test_file)


    ##### Adamic-Adar Index Scoring #####
    aa_index = adamic_adar_index(train_graph, test_list)
    # write_top_links(find_top_links(test_list, aa_index))
    # write_full_results(label_top_links(test_list, aa_index))


    ##### GCN Link Prediction #####
    # Create pairs of data and get features for each node
    train_pairs = create_labelled_pairs(train_graph)
    features_dict = extract_node_features(train_graph)

    # All features as a matrix
    all_features = []
    for key, value in features_dict.items():
        all_features.append(value)
    all_features = torch.tensor(all_features)

    node_embeddings = get_node_embeddings(train_graph, all_features)
    X_train, y_train, X_test, y_test = create_datasets(node_embeddings, train_pairs)

    # Set up model
    link_pred_model = LinkPredictionModel(embedding_size=10, hidden_size1=32, hidden_size2=16, hidden_size3=16, hidden_size4=8, hidden_size5=4, output_size=1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(link_pred_model.parameters(), lr=0.001)

    # Train/load model - IMPORTANT: Set load_weight_path="" if training from scratch
    model, train_loss_hist, test_loss_hist = train_model(link_pred_model, optimizer, criterion, 5000, X_train, y_train, X_test, y_test, load_weight_path="link_pred_model_weights.pth")

    # Predicting
    predictions = model_predict(model, test_list, node_embeddings)
    write_full_results(label_top_links(test_list, predictions), file_name="full_results_GCN.csv")


    ##### Adamic-Adar Index Combined with GCN #####
    # IMPORTANT: Uncommect all GCN code and the first line for Adamic-Adar Index to obtain combined results
    write_full_results(label_top_links(test_list, (torch.tensor(aa_index).unsqueeze(1) + predictions) / 2), file_name="full_results_AA_GCN.csv")
