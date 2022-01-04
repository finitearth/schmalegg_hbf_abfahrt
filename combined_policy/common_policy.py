import torch.nn as nn
from torch_geometric.nn import Linear, BatchNorm, global_add_pool, SAGEConv
import torch.nn.functional as F
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ValueNet(nn.Module):
    def __init__(self, config):
        super(ValueNet, self).__init__()
        self.config = config
        hidden_neurons = config.hidden_neurons
        self.lins = [Linear(hidden_neurons, hidden_neurons).to(device) for _ in range(config.n_lin_value)]
        self.lin1 = Linear(hidden_neurons, 1)
        self.bn = BatchNorm(hidden_neurons)

        self.conv1 = SAGEConv(config.n_node_features, config.hidden_neurons, normalize=config.normalize, aggr=config.aggr_con)
        self.conv2 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=config.normalize, aggr=config.aggr_con)
        self.conv3 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=False, aggr=config.aggr_dest)
        self.conv4 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=config.normalize, aggr=config.aggr_con)
        self.conv5 = SAGEConv(config.hidden_neurons, config.hidden_neurons, normalize=config.normalize,
                          aggr=config.aggr_con)

    def forward(self, x, eic, eid, eit, batch):
        x = self.conv1(x, eid)
        for _ in range(self.config.it_b4_dest):
            x = self.conv2(x, eic)
        x = self.conv3(x, eit)
        x = self.conv4(x, eid)
        for _ in range(self.config.it_aft_dest):
            x = self.conv5(x, eic)
        x = global_add_pool(x, batch)
        # x = self.bn(x)
        for lin in self.lins:
            x = lin(x)
            x = F.relu(x)
        x = self.lin1(x)

        return x
