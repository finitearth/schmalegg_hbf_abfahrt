import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import numpy as np

NODE_FEATURES = 10
OUTPUT_FEATURES = 4


class CustomNet(nn.Module):
    def __init__(self, edge_index, *args, **kwargs):
        super(CustomNet, self).__init__()
        self.edge_index = torch.tensor(edge_index)
        self.value_net = ValueNet()
        self.policy_net = PolicyNet(edge_index)
        self.latent_dim_pi = 20
        self.latent_dim_vf = 20
       # self.forward_actor = self.policy_net

    def forward(self, x, use_sde=False):
        x_p = self.policy_net(x)
        # print("x_v requested from netsforreal")
        #x_v = self.value_net(x_p)
        return x_p, x_p


class PolicyNet(nn.Module):
    def __init__(self, edge_index, *args, **kwargs):
        super(PolicyNet, self).__init__()
        self.edge_index = torch.tensor(edge_index)

        self.conv1 = GCNConv(NODE_FEATURES, 16, self.edge_index)
        self.conv2 = GCNConv(16, OUTPUT_FEATURES, self.edge_index)

    def forward(self, x, use_sde=False):
        #print(x.size())
        if x.size(-1) == 20:
            return x
        #+
        # try:
        #     if x.size() <= torch.Size([50]):

        #
        #     else:
        #         x = torch.reshape(x, (64, 5, 10))
        # except Exception as _:
        #     print(x)
        #     raise _

        # if x.size() != torch.Size([2, 20]):

        y = torch.tensor([])
        for x_ in x:
            x_ = torch.reshape(x_, (5, 10))
            x_ = self.conv1(x_)
            x_ = F.relu(x_)
            x_ = self.conv2(x_)

            x_ = torch.reshape(x_, (20,))
            try:
                y = torch.stack((y, x_))
            except Exception as _:
                y = x_

        #print(y)
        x = torch.Tensor(y)#, dtype=torch.float32)
    # elif x.size() == torch.Size([2, 20]):
    #     return x

        return x


class ValueNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ValueNet, self).__init__()

        self.linear = nn.Linear(OUTPUT_FEATURES * 5, 1)

    def forward(self, x, use_sde=False):
        x = self.linear(x)
        return x


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_index, *args, **kwargs):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_index = edge_index

    def forward(self, x, use_sde=False):
        # Step 1: Add self-loops
        #        edge_index, _ = add_self_loops(edge_index, num_nodes=5) #x.size(0)) 5 Bahnhöfe

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(self.edge_index, size=(x.size(0), x.size(0)), x=x)
