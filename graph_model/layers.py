import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.utils import (add_remaining_self_loops, degree,
                                   remove_self_loops)
from torch_scatter import scatter_add

from utils.inits import glorot, reset, zeros
from utils.normalization import NeighborNorm, SingleNorm


class SAGELayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, norm='None'):
        super().__init__()
        self.norm = norm
        assert self.norm in ['neighbornorm', 'None']
        self.lin_l = torch.nn.Linear(in_dim, out_dim)
        self.lin_r = torch.nn.Linear(in_dim, out_dim)
        if self.norm == 'neighbornorm':
            self.normlayer_l = NeighborNorm(in_dim)
            self.normlayer_r = SingleNorm(in_dim)
        else:
            self.batchnorm = torch.nn.BatchNorm1d(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        zeros(self.lin_r.bias)
        zeros(self.lin_r.bias)
        if self.norm == 'neighbornorm':
            self.normlayer_l.reset_parameters()
            self.normlayer_r.reset_parameters()

    def forward(self, x, edge_index):
        row, col = edge_index
        if self.norm == 'neighbornorm':
            x_j = self.normlayer_l(x, edge_index)
        else:
            x_j = x[col]

        out = scatter_add(src=x_j, index=row, dim=0, dim_size=x.size(0))

        if self.norm == 'neighbornorm':
            out = self.lin_l(out) + self.lin_r(self.normlayer_r(x))
        else:
            out = self.lin_l(out) + self.lin_r(x)

        if self.norm == 'neighbornorm':
            out = F.relu(out)
        else:
            out = self.batchnorm(F.relu(out))
        return out


class GCNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, norm='None'):
        super().__init__()
        self.norm = norm

        self.linear = torch.nn.Linear(in_dim, out_dim)

        assert self.norm in ['neighbornorm', 'None']
        if self.norm == 'neighbornorm':
            self.normlayer = NeighborNorm(in_dim)
        else:
            self.normlayer = torch.nn.BatchNorm1d(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear.weight)
        zeros(self.linear.bias)
        if self.norm == 'neighbornorm':
            self.normlayer.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)

        row, col = edge_index
        deg = degree(row)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.norm == 'neighbornorm':
            x_j = self.normlayer(x, edge_index)
        else:
            x_j = x[col]

        x_j = norm.view(-1, 1) * x_j
        out = scatter_add(src=x_j, index=row, dim=0, dim_size=x.size(0))

        if self.norm == 'neighbornorm':
            out = F.relu(self.linear(out))
        else:
            out = self.normlayer(F.relu(self.linear(out)))

        return out


class GINLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, norm='None'):
        super().__init__()
        self.norm = norm
        assert self.norm in ['neighbornorm', 'None']

        self.mlp = Sequential(Linear(in_dim, out_dim), ReLU(),
                              Linear(out_dim, out_dim), ReLU(),
                              BatchNorm1d(out_dim))

        if self.norm == 'neighbornorm':
            self.normlayer_l = NeighborNorm(in_dim)
            self.normlayer_r = SingleNorm(in_dim)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)
        if self.norm == 'neighbornorm':
            self.normlayer_l.reset_parameters()
            self.normlayer_r.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index

        if self.norm == 'neighbornorm':
            x_j = self.normlayer_l(x, edge_index)
        else:
            x_j = x[col]

        out = scatter_add(src=x_j, index=row, dim=0, dim_size=x.size(0))

        if self.norm == 'neighbornorm':
            out = self.mlp(self.normlayer_r(x) + out)
        else:
            out = self.mlp(x + out)
        return out
