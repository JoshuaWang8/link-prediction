import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from feature_extraction import get_adjacency_matrix
from sklearn.model_selection import train_test_split


def get_node_embeddings(graph, node_features):
    """
    Gets embeddings for all nodes in graph.

    Parameters:
        graph: Graph for which features will be obtained.
        node_features: Features for all nodes as nested lists.

    Returns:
        Embeddings: Nested lists of embeddings for each node.
    """
    adjacency_matrix = get_adjacency_matrix(graph)
    adjacency_matrix = adjacency_matrix + np.identity(adjacency_matrix.shape[0]) # Add self-loops

    # Inverse degree matrix for normalisation
    D = np.array(np.sum(adjacency_matrix, axis=0))
    D = np.matrix(np.diag(D))

    adjacency_matrix = D**-1 * adjacency_matrix

    return F.relu(torch.matmul(torch.tensor(adjacency_matrix), node_features))


def create_datasets(node_embeddings, train_pairs):
    """
    Creates datasets using node embeddings from the trained pairs.

    Parameters:
        node_embeddings: Nested lists of embeddings for each node.
        train_pairs: Paired nodes to train on.

    Returns:
        X_train: Embeddings for training as a PyTorch tensor.
        y_train: Labels for training as a PyTorch tensor.
        X_test: Embeddings for testing as a PyTorch tensor.
        y_test: Labels for testing as a PyTorch tensor.
    """
    pair_embeddings = []
    pair_labels = []

    for i in range(len(train_pairs)):
        concat_embeddings = torch.cat([node_embeddings[train_pairs.loc[i]["node1"]], node_embeddings[train_pairs.loc[i]["node2"]]])
        pair_embeddings.append(concat_embeddings)
        pair_labels.append(torch.tensor(train_pairs.loc[i]["label"]))

    pair_embeddings = torch.stack(pair_embeddings).float()
    pair_labels = torch.stack(pair_labels).float()

    # Convert tensors to numpy arrays
    pair_embeddings_np = pair_embeddings.numpy()
    pair_labels_np = pair_labels.numpy()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(pair_embeddings_np, pair_labels_np, test_size=0.2, random_state=8)

    # Convert back to tensors
    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    return X_train, y_train, X_test, y_test


def train_model(model, optimizer, criterion, num_epochs, X_train, y_train, X_val, y_val, load_weight_path="link_pred_model_weights.pth"):
    """
    Trains the model. If the load_weight_path parameter is set to an empty string (""), model will be trained from scratch. Otherwise,
    model weights will be loaded.

    Parameters:
        model: Model for training.
        optimizer: Optimizer for model.
        criterion: Loss function for model.
        num_epochs: Number of epochs to train for.
        X_train: Embeddings for training as a PyTorch tensor.
        y_train: Labels for training as a PyTorch tensor.
        X_val: Embeddings for testing as a PyTorch tensor.
        y_val: Labels for testing as a PyTorch tensor.
        load_weight_path: Path from which weights of the pretrained model will be loaded from. If "", then model will be trained from scratch.

    Returns:
        model: Trained PyTorch model object.

    """
    train_loss_hist = []
    val_loss_hist = []

    if load_weight_path == "":
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            output_train = model(X_train)
            loss_train = criterion(output_train.squeeze(), y_train)
            loss_train.backward()
            optimizer.step()

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                output_test = model(X_val)
                loss_test = criterion(output_test.squeeze(), y_val)

            train_loss_hist.append(loss_train.item())
            val_loss_hist.append(loss_test.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_train.item()}, Test Loss: {loss_test.item()}')
    else:
        model = LinkPredictionModel(embedding_size=10, hidden_size1=32, hidden_size2=16, hidden_size3=16, hidden_size4=8, hidden_size5=4, output_size=1)
        model.load_state_dict(torch.load(load_weight_path))
        model.eval()

    return model, train_loss_hist, val_loss_hist


def model_predict(model, test_list, node_embeddings):
    """
    Predicts using the model.

    Parameters:
        model: Model for predicting.
        test_list: List of pairs of nodes to predict on.
        node_embeddings: Embeddings of each node.

    Returns:
        test_output: Predicted probabilities of the node pair having a link (between 0 and 1).
    """
    with torch.no_grad():
        test_examples = []
        for i in range(len(test_list)):
            node1_embedding = node_embeddings[test_list[i][0]]
            node2_embedding = node_embeddings[test_list[i][1]]
            concatenated_embeddings = torch.cat([node1_embedding, node2_embedding])
            test_examples.append(concatenated_embeddings)

        test_examples = torch.stack(test_examples).float()

        model.eval()
        test_output = model(test_examples)
        
        return test_output


class LinkPredictionModel(nn.Module):
    def __init__(self, embedding_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
        super(LinkPredictionModel, self).__init__()
        self.fc1 = nn.Linear(embedding_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, node_pair_embeddings):
        x = F.relu(self.fc1(node_pair_embeddings))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x
