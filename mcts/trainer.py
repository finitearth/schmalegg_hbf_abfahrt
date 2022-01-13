import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

import utils
from mcts import MCTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, value_net, policy_net, config):
        self.value_net = value_net.to(device)
        self.policy_net = policy_net.to(device)
        self.config = config
        self.pi_examples = []
        self.v_examples = []
        self.mcts = MCTS()

        self.pi_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr_pi)
        self.v_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.config.lr_v)
        # self.lr_pi = ExponentialLR(optimizer=self.pi_optim, gamma=.995)
        # self.lr_v = ExponentialLR(optimizer=self.v_optim, gamma=.995)

    def execute_episode(self, root):
        root = self.mcts.search(root)
        pi_examples = []
        v_examples = []
        node = root
        while not node.is_leaf():
            best_child = node.get_best_child()
            if node.observation is not None:
                input, eic, eit, eid, batch = node.observation
                actions = node.possible_actions
                best_action = best_child.action
                # one_hot = torch.where(actions[:, :]==best_action[None,:], True, False).all(dim=-1).flatten(start_dim=-2)*1.
                target = (actions == best_action).all(dim=-1).nonzero(as_tuple=True)[0]
                pi_example = PiData(x=input, c_edge_index=eic, t_edge_index=eit, d_edge_index=eid, actions=actions,
                                    target=target)
                pi_examples.append(pi_example)

                value = node.value_sum
                # print(node.value_sum)
                v_example = VData(x=input, c_edge_index=eic, t_edge_index=eit, d_edge_index=eid, target=value)
                v_examples.append(v_example)

                node = best_child

        return pi_examples, v_examples

    def train(self, pi_examples, v_examples):
        pi_data_loader = DataLoader(pi_examples, batch_size=256, shuffle=True, collate_fn=utils.collate) #self.config.batch_size_pi
        v_data_loader = PyGDataLoader(v_examples, batch_size=256, shuffle=True)  # self.config.batch_size_v
        print(f"Batchcount: {len(v_data_loader)}")
        l_pi_function = nn.CrossEntropyLoss()
        l_v_function = nn.MSELoss()
        n_epochs = 8  # self.config.n_epochs):
        for epoch in range(n_epochs):
            pi_losses, v_losses, pi_acc, v_expvar, batches = [], [], [], [], []
            for x in iter(pi_data_loader):
                x = x.to(device)
                input, eic, eid, eit, actions, target_pis, batch = x.x, x.c_edge_index, x.d_edge_index, x.t_edge_index, x.actions, x.target, x.batch
                pred_pis = self.policy_net(actions, input, eic, eit, eid, batch)
                l_pi = l_pi_function(pred_pis, target_pis)
                pi_losses.append(l_pi)
                l_pi.backward()
                pred_onehot = torch.argmax(pred_pis, -1)
                pi_acc.append(sum(pred_onehot == target_pis) / len(pred_onehot))
                batches.append(max(batch)+1)
                self.pi_optim.step()
                # self.lr_pi.step()

            for x in iter(v_data_loader):
                x = x.to(device)
                input, eic, eid, eit, target_vs, batch = x.x, x.c_edge_index, x.d_edge_index, x.t_edge_index, x.target, x.batch
                pred_vs = self.value_net(input, eic, eit, eid, batch)
                pred_vs = pred_vs.squeeze(1)
                l_v = l_v_function(pred_vs, target_vs.float())
                v_losses.append(l_v)
                l_v.backward()
                exp_var = 1 - torch.var(target_vs-pred_vs) / (torch.var(target_vs)+1e-6)
                v_expvar.append(exp_var)
                self.v_optim.step()
                # self.lr_v.step()

            print(f"Epoch {epoch + 1}/{n_epochs}, "
                  f"v_loss: {sum(v_losses) / len(v_losses):.3f},"
                  f" pi_loss: {sum(pi_losses) / len(pi_losses):.3f},"
                  f" pi_acc: {sum(pi_acc) / len(pi_acc) * 100:.3f}% , "
                  f"v_expvar: {sum(v_expvar) / len(v_expvar):.4f}, "
                  f"max batch: {max(batches)}")


class PiData(Data):
    def __init__(self, x=None, c_edge_index=None, d_edge_index=None, t_edge_index=None, actions=None, target=None,
                 **kwargs):
        super().__init__(x=x, target=target, actions=actions,
                         c_edge_index=c_edge_index, d_edge_index=d_edge_index, t_edge_index=t_edge_index, **kwargs)
        if c_edge_index is not None:
            c_edge_index, _ = add_self_loops(c_edge_index)


class VData(Data):
    def __init__(self, x=None, c_edge_index=None, d_edge_index=None, t_edge_index=None, target=None, **kwargs):
        super().__init__(x=x, target=target, **kwargs)
        if c_edge_index is not None:
            c_edge_index, _ = add_self_loops(c_edge_index)
        if c_edge_index is not None:
            self.c_edge_index = c_edge_index.long()  # if c_edge_index else None
            self.d_edge_index = d_edge_index.long()  # if d_edge_index else None
            self.t_edge_index = t_edge_index.long()  # if t_edge_index else None
            self.target = target