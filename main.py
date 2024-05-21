import torch
from utilities import *
from neighbourhood_prediction import *
from feature_extraction import *
from random_walk_prediction import *
from gcn import *

import networkx as nx


def GCN_scoring(train_graph, test_list):
    """
    Performs link prediction using GCN and binary classification.
    """
    # Create pairs of data and get features for each node
    train_pairs = create_labelled_pairs(train_graph)
    features_dict = extract_node_features(train_graph)

    # Get embeddings and create train/test sets
    node_embeddings = get_node_embeddings(train_graph, features_dict)
    X_train, y_train, X_test, y_test = create_datasets(node_embeddings, train_pairs)

    # Set up model
    link_pred_model = LinkPredictionModel(embedding_size=10, hidden_size1=32, hidden_size2=16, hidden_size3=16, hidden_size4=8, hidden_size5=4, output_size=1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(link_pred_model.parameters(), lr=0.001)

    # Train/load model - IMPORTANT: Set load_weight_path="" if training from scratch
    model, _, _ = train_model(link_pred_model, optimizer, criterion, 5000, X_train, y_train, X_test, y_test, load_weight_path="link_pred_model_weights_concat.pth")

    # Predicting
    predictions = model_predict(model, test_list, node_embeddings)
    
    return predictions


def AA_GCN_scoring(train_graph, test_list):
    """
    Performs link prediction using the average score between GCN and Adamic-Adar Index.
    """
    # Create pairs of data and get features for each node
    train_pairs = create_labelled_pairs(train_graph)
    features_dict = extract_node_features(train_graph)

    # Get embeddings and create train/test sets
    node_embeddings = get_node_embeddings(train_graph, features_dict)
    X_train, y_train, X_test, y_test = create_datasets(node_embeddings, train_pairs)

    # Set up model
    link_pred_model = LinkPredictionModel(embedding_size=10, hidden_size1=32, hidden_size2=16, hidden_size3=16, hidden_size4=8, hidden_size5=4, output_size=1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(link_pred_model.parameters(), lr=0.001)

    # Train/load model - IMPORTANT: Set load_weight_path="" if training from scratch
    model, _, _ = train_model(link_pred_model, optimizer, criterion, 5000, X_train, y_train, X_test, y_test, load_weight_path="link_pred_model_weights_concat.pth")

    # Predicting
    predictions = model_predict(model, test_list, node_embeddings)
    aa_index = adamic_adar_index(train_graph, test_list)

    return (torch.tensor(aa_index).unsqueeze(1) + predictions) / 2


if __name__ == "__main__":
    training_file = "./data/trainingset.csv"
    test_file = "./data/testset.csv"

    train_graph = load_graph(training_file)
    test_list = load_test_set_as_list(test_file)


    # ##### GCN Link Scoring ##### - 0.85300
    scores = GCN_scoring(train_graph, test_list)

    # ##### Adamic-Adar Index Combined with GCN ##### - 0.90500
    # scores = AA_GCN_scoring(train_graph, test_list)

  
    # ##### Jaccard Similarity Scoring ##### - 0.90300
    # scores = jaccard_similarity(train_graph, test_list)

    # ##### Cosine Similarity Scoring ##### - 0.90300
    # scores = cosine_similarity(train_graph, test_list)

    # ##### Preferential Attachment Scoring #####  0.84700
    # scores = preferential_attachment(train_graph, test_list)

    # ##### Adamic-Adar Index Scoring ##### - 0.90300
    # scores = adamic_adar_index(train_graph, test_list)

    # ##### Katz Measure Scoring ##### - 0.83300
    # scores = katz_measure(train_graph, test_list)


    ##### Creating restuls files #####
    # write_top_links(find_top_links(test_list, scores))
    write_full_results(label_top_links(test_list, scores),  file_name="test.csv")