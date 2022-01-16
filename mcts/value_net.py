import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, Linear, BatchNorm, global_add_pool, GCNConv


class ValueNet(nn.Module):
    def __init__(self, config):
        super(ValueNet, self).__init__()
        self.config = config
        hidden_neurons = config.hidden_neurons
        self.lins = Sequential('x', [(Linear(hidden_neurons, hidden_neurons), 'x->x')
                                     for _ in range(2)])
        self.lin1 = Linear(hidden_neurons, 1)

        self.conv1 = GCNConv(config.n_node_features, config.hidden_neurons, normalize=False, aggr=config.aggr_con)
        self.conv2 = GCNConv(config.hidden_neurons, config.hidden_neurons, normalize=False, aggr=config.aggr_con)
        self.conv3 = GCNConv(config.hidden_neurons, config.hidden_neurons, normalize=False, aggr=config.aggr_dest)

    def forward(self, x, adj, adj_attr, pass_adj, pass_adj_attr,  train_adj, train_adj_attr, batch):
        x = self.conv1(x, train_adj, edge_weight=train_adj_attr)
        x = self.conv2(x, adj, edge_weight=adj_attr)
        x = self.conv3(x, pass_adj, edge_weight=pass_adj_attr)
        x = global_add_pool(x, batch)
        # x = self.bn(x)
        x = self.lins(x)
        x = self.lin1(x)

        return x
