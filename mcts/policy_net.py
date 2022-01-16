import torch
import torch.nn as nn
from torch_geometric.nn import Linear, Sequential
from torch_geometric.utils import to_dense_batch


class PolicyNet(nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        hidden_neurons = config.hidden_neurons
        convclass = config.conv
        self.conv1 = convclass(config.n_node_features, config.hidden_neurons, aggr=config.aggr_con)
        self.conv2 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv3 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.lin2 = Linear(hidden_neurons, hidden_neurons)

        self.conv4 = convclass(hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv5 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv6 = convclass(config.hidden_neurons,
                               config.hidden_neurons, aggr=config.aggr_con)
        self.lin3 = Linear(hidden_neurons, hidden_neurons)

        self.conv7 = convclass(hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv8 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.conv9 = convclass(config.hidden_neurons, config.hidden_neurons, aggr=config.aggr_con)
        self.lin4 = Linear(hidden_neurons, hidden_neurons)

        self.lins = Sequential('x', [(Linear(hidden_neurons, hidden_neurons), 'x->x')
                                     for _ in range(2)])
        self.lin1 = Linear(hidden_neurons, config.action_vector_size)
        self.softmax = nn.Softmax(dim=1)
        self.config = config
        self.n = config.action_vector_size // 2
        self.dest_vecs = None
        self.start_vecs = None

    def forward(self, actions, x, adj, adj_attr, pass_adj, pass_adj_attr,  train_adj, train_adj_attr, batch):
        x = self.conv1(x, train_adj, train_adj_attr)
        x = torch.relu(x)
        x = self.conv2(x, adj, adj_attr)
        x = torch.relu(x)
        x = self.conv3(x, pass_adj, pass_adj_attr)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)

        # x = self.conv3(x, train_adj, train_adj_attr)
        # x = torch.relu(x)
        # x = self.conv4(x, adj, train_adj_attr)
        # x = torch.relu(x)
        # x = self.conv5(x, pass_adj, pass_adj_attr)
        # x = torch.relu(x)
        # x = self.lin3(x)
        # x = torch.relu(x)

        x = self.lins(x)
        x = self.lin1(x)
        x, _ = to_dense_batch(x, batch)
        start_vecs = x[:, :, :self.n]
        dest_vecs = x[:, :, self.n:]

        starts = start_vecs[:, actions[..., 0]]#
        dests = dest_vecs[:, actions[..., 1]]

        probs = torch.einsum('bijk,bijk->bi', starts, dests)
        probs = self.softmax(probs)

        return probs
