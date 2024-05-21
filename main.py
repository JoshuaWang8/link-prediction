import torch
from utilities import *
from neighbourhood_prediction import *
from feature_extraction import *
from random_walk_prediction import *
from gcn import *
from embedding_distances import *


def GCN_scoring(node_embeddings, test_list, train_pairs):
    """
    Performs link prediction using GCN and binary classification.
    """
    # Get embeddings and create train/test sets
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


if __name__ == "__main__":
    # Loading in training and testing data
    training_file = "./data/trainingset.csv"
    test_file = "./data/testset.csv"

    train_graph = load_graph(training_file)
    test_list = load_test_set_as_list(test_file)


    ##### Jaccard Similarity Scoring ##### - 0.90300
    # jaccard_scores = jaccard_similarity(train_graph, test_list)

    ##### Cosine Similarity Scoring ##### - 0.90300
    # cosine_scores = cosine_similarity(train_graph, test_list)

    ##### Preferential Attachment Scoring #####  0.84700
    # pref_attach_scores = preferential_attachment(train_graph, test_list)

    ##### Adamic-Adar Index Scoring ##### - 0.90300
    # aa_scores = adamic_adar_index(train_graph, test_list)

    ##### Katz Measure Scoring ##### - 0.83300
    # katz_scores = katz_measure(train_graph, test_list)


    # Create pairs of data and get features for each node (NECESSARY FOR METHODS BELOW)
    train_pairs = create_labelled_pairs(train_graph)
    features_dict = extract_node_features(train_graph)
    node_embeddings = get_node_embeddings(train_graph, features_dict)

    ##### GCN Link Scoring ##### - 0.85300
    # gcn_scores = GCN_scoring(node_embeddings, test_list, train_pairs)

    ##### Embedding Distance Comparison ##### - 0.82700
    embed_dist_scores = embedding_distance_scores(node_embeddings, test_list)

    ##### Adamic-Adar Index Combined with GCN ##### - 0.90500
    # gcn_aa_scores = (gcn_scores + torch.tensor(aa_scores).unsqueeze(1))/2


    ##### Creating results files - change scoring method as required #####
    # write_top_links(find_top_links(test_list, scores))
    write_full_results(label_top_links(test_list, embed_dist_scores),  file_name="test.csv")