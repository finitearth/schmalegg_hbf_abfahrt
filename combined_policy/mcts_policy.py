import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear


class PolicyNet(nn.Module):
    def __init__(self, hidden_neurons, output_features, config):
        super(PolicyNet, self).__init__()
        self.lins = [Linear(hidden_neurons, hidden_neurons) for _ in range(config.n_lin_policy)]
        self.lin1 = Linear(hidden_neurons, output_features)
        self.config = config
        self.n = config.actionvector_size // 2
        self.dest_vecs = None
        self.start_vecs = None

    def calc_probs(self, x, edge_index_connections, edge_index_destinations):
        x = self.conv1(x, edge_index_connections)
        x = self.activation(x)
        for _ in range(self.config.it_b4_dest):
            x = self.conv2(x, edge_index_connections)
            x = self.activation(x)
        x = self.conv3(x, edge_index_destinations)
        x = self.activation(x)

        for _ in range(self.config.it_aft_dest):
            x = self.conv4(x, edge_index_connections)
            x = self.activation(x)

        for lin in self.lins:
            x = lin(x)
        x = self.lin1(x)

        self.dest_vecs = x[:, :, self.n:]
        self.start_vecs = x[:, :, :self.n]

    @staticmethod
    def get_prob(self, actions, start_vecs, dest_vecs):
        starts = start_vecs[:, :, actions[:, 0]] # batches, action, stations, bool_starting
        dests = dest_vecs[:, :, actions[:, 1]]
        probs = starts @ dests
        probs = F.softmax(probs, dim=1)

        return probs


