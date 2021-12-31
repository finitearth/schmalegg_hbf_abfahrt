import torch.nn as nn
from torch_geometric.nn import Linear, BatchNorm, global_add_pool
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, hidden_neurons, config):
        super(ValueNet, self).__init__()
        self.lins = [Linear(hidden_neurons, hidden_neurons) for _ in range(config.n_lin_value)]
        self.lin1 = Linear(hidden_neurons, 1)
        self.bn = BatchNorm(hidden_neurons, training=False)

    def forward(self, x, edge_index_connections, edge_index_destinations, batch):
        x = global_add_pool(x, batch)
        x = self.bn(x, training=False)
        for lin in self.lins:
            x = lin(x)
            x = F.relu(x)
        x = self.lin1(x)

        return x
