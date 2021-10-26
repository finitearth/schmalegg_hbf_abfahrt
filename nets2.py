from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        hidden_channels = 128
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights

        x = self.lin1(x)
        x = self.lin2(x)


        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    

class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dataset.num_node_features)
        # self.conv2 = GCNConv(dataset.num_node_features, dataset.num_classes)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for _ in range(2):
            x = self.conv1(x, edge_index)
       # x = F.relu(x)
        #x = F.dropout(x, training=self.training)
       # x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def plot_dataset(dataset):
    edges_raw = dataset.data.edge_index.numpy()
    edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
    labels = dataset.data.y.numpy()

    G = nx.Graph()
    G.add_nodes_from(list(range(np.max(edges_raw))))
    G.add_edges_from(edges)
    plt.subplot(111)
    options = {
        'node_size': 30,
        'width': 0.2,
    }
    nx.draw(G, with_labels=False, node_color=labels.tolist(), cmap=plt.cm.tab10, font_weight='bold', **options)
    plt.show()


def test(data, train=True):
    model.eval()

    correct = 0
    pred = model(data).max(dim=1)[1]


    if train:
        correct += pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
        return correct / (len(data.y[data.train_mask]))
    else:
        correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        return correct / (len(data.y[data.test_mask]))


def train(data, plot=False):
    train_accuracies, test_accuracies = list(), list()
    for epoch in tqdm(range(100), disable=True):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_acc = test(data)
        test_acc = test(data, train=False)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
               format(epoch, loss, train_acc, test_acc))


    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Validation accuracy")
        plt.xlabel("# Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # plot_dataset(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print("--- Starting Training ---")
    train(data, plot=True)
    #print(len(model.state_dict()))
