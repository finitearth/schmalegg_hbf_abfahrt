import torch.nn as nn
from torch_geometric.nn.dense import Linear


class PolicyNet(nn.Module):
    def __init__(self, hidden_neurons, output_features, config):
        super(PolicyNet, self).__init__()
        self.lins = [Linear(hidden_neurons, hidden_neurons) for _ in range(config.n_lin_policy)]
        self.lin1 = Linear(hidden_neurons, output_features)

    def forward(self, x, edge_index_connections, edge_index_destinations):
        for lin in self.lins:
            x = lin(x)
        x = self.lin1(x)
        return x
