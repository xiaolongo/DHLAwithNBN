import torch
import torch.nn.functional as F
from torch_geometric.utils import (add_remaining_self_loops, add_self_loops,
                                   degree, dropout_adj, remove_self_loops,
                                   softmax)
from torch_scatter import scatter, scatter_add

from utils.inits import glorot, zeros
from utils.normalization import NeighborNorm, NodeNorm, PairNorm, SingleNorm


class GCNLayer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout=False,
                 norm='None',
                 residual=False,
                 activation=True):
        super().__init__()
        self.dropout = dropout
        self.norm = norm
        self.residual = residual
        self.activation = activation

        self.linear = torch.nn.Linear(in_dim, out_dim)

        assert self.norm in [
            'batchnorm', 'layernorm', 'dropedge', 'pairnorm', 'nodenorm',
            'neighbornorm', 'None'
        ]
        if self.norm == 'batchnorm':
            self.normlayer = torch.nn.BatchNorm1d(out_dim)
        elif self.norm == 'layernorm':
            self.normlayer = torch.nn.LayerNorm(out_dim)
        elif self.norm == 'neighbornorm':
            self.normlayer = NeighborNorm(in_dim)
        elif self.norm == 'pairnorm':
            self.normlayer = PairNorm()
        elif self.norm == 'nodenorm':
            self.normlayer = NodeNorm()
        else:
            self.normlayer = None

        if in_dim != out_dim:
            self.residual = False

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.linear.weight)
        zeros(self.linear.bias)

    def forward(self, x, edge_index):
        x_in = x
        edge_index, _ = add_remaining_self_loops(edge_index)

        if self.norm == 'dropedge':
            if self.training:
                edge_index, _ = dropout_adj(edge_index,
                                            force_undirected=True,
                                            training=True)
            else:
                edge_index, _ = dropout_adj(edge_index,
                                            force_undirected=True,
                                            training=False)

        row, col = edge_index
        deg = degree(row)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # x = self.linear(x)

        if self.norm == 'neighbornorm':
            x_j = self.normlayer(x, edge_index)
        else:
            x_j = x[col]

        x_j = norm.view(-1, 1) * x_j
        out = scatter_add(src=x_j, index=row, dim=0, dim_size=x.size(0))

        out = self.linear(out)

        if self.activation:
            out = F.relu(out)

        if self.norm == 'batchnorm':
            out = self.normlayer(out)
        elif self.norm == 'layernorm':
            out = self.normlayer(out)
        elif self.norm == 'pairnorm':
            out = self.normlayer(out)
        elif self.norm == 'nodenorm':
            out = self.normlayer(out)

        if self.residual:
            out = x_in + out

        if self.dropout:
            out = F.dropout(out, p=0.5, training=self.training)

        return out


class SAGELayer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout=False,
                 norm='None',
                 residual=False,
                 activation=True):
        super().__init__()
        self.dropout = dropout
        self.norm = norm
        self.residual = residual
        self.activation = activation

        self.lin_l = torch.nn.Linear(in_dim, out_dim)
        self.lin_r = torch.nn.Linear(in_dim, out_dim)

        assert self.norm in [
            'batchnorm', 'layernorm', 'dropedge', 'pairnorm', 'nodenorm',
            'neighbornorm', 'None'
        ]

        if self.norm == 'batchnorm':
            self.normlayer = torch.nn.BatchNorm1d(out_dim)
        elif self.norm == 'layernorm':
            self.normlayer = torch.nn.LayerNorm(out_dim)
        elif self.norm == 'neighbornorm':
            self.normlayer_l = NeighborNorm(in_dim)
            self.normlayer_r = SingleNorm(in_dim)
        elif self.norm == 'pairnorm':
            self.normlayer = PairNorm()
        elif self.norm == 'nodenorm':
            self.normlayer = NodeNorm()
        else:
            self.normlayer = None

        if in_dim != out_dim:
            self.residual = False

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        zeros(self.lin_l.bias)
        zeros(self.lin_r.bias)

    def forward(self, x, edge_index):
        x_in = x
        row, col = edge_index

        if self.norm == 'neighbornorm':
            x_j = self.normlayer_l(x, edge_index)
        else:
            x_j = x[col]

        out = scatter(src=x_j,
                      index=row,
                      dim=0,
                      dim_size=x.size(0),
                      reduce='mean')

        if self.norm == 'neighbornorm':
            out = self.lin_l(out) + self.lin_r(self.normlayer_r(x))
        else:
            out = self.lin_l(out) + self.lin_r(x)

        if self.activation:
            out = F.relu(out)

        if self.norm == 'batchnorm':
            out = self.normlayer(out)
        elif self.norm == 'layernorm':
            out = self.normlayer(out)
        elif self.norm == 'pairnorm':
            out = self.normlayer(out)
        elif self.norm == 'nodenorm':
            out = self.normlayer(out)

        if self.residual:
            out = x_in + out

        if self.dropout:
            out = F.dropout(out, p=0.5, training=self.training)

        return out


class GATLayer(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 heads=4,
                 concat=True,
                 dropout=False,
                 norm='None',
                 residual=True,
                 activation=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        self.dropout = dropout
        self.norm = norm
        self.residual = residual
        self.activation = activation

        assert self.norm in [
            'batchnorm', 'layernorm', 'dropedge', 'pairnorm', 'nodenorm',
            'neighbornorm', 'None'
        ]

        if self.norm == 'batchnorm':
            self.normlayer = torch.nn.BatchNorm1d(heads * out_dim)
        elif self.norm == 'layernorm':
            self.normlayer = torch.nn.LayerNorm(heads * out_dim)
        elif self.norm == 'neighbornorm':
            self.normlayer_l = NeighborNorm(heads * out_dim)
            self.normlayer_r = SingleNorm(heads * out_dim)
        elif self.norm == 'pairnorm':
            self.normlayer = PairNorm()
        elif self.norm == 'nodenorm':
            self.normlayer = NodeNorm()
        else:
            self.normlayer = None

        self.lin = torch.nn.Linear(in_dim, heads * out_dim, bias=False)
        if concat:
            self.lin_residual = torch.nn.Linear(in_dim, heads * out_dim)
        else:
            self.lin_residual = torch.nn.Linear(in_dim, out_dim)

        self.att_i = torch.nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.att_j = torch.nn.Parameter(torch.Tensor(1, heads, out_dim))

        if concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_dim))
        else:
            self.bias = torch.nn.Parameter(torch.Tensor(out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.lin_residual.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.lin_residual.bias)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x_in = x
        x = self.lin(x)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if self.norm == 'dropedge':
            if self.training:
                edge_index, _ = dropout_adj(edge_index,
                                            force_undirected=True,
                                            training=True)
            else:
                edge_index, _ = dropout_adj(edge_index,
                                            force_undirected=True,
                                            training=False)

        row, col = edge_index

        x_i = x[row]

        if self.norm == 'neighbornorm':
            x_j = self.normlayer_l(x, edge_index)
        else:
            x_j = x[col]

        x_i = x_i.view(-1, self.heads, self.out_dim)
        x_j = x_j.view(-1, self.heads, self.out_dim)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2, inplace=True)
        alpha = softmax(alpha, row, num_nodes=x.size(0))

        alpha = F.dropout(alpha, p=0.5, training=self.training)

        x_j = x_j * alpha.view(-1, self.heads, 1)

        out = scatter_add(src=x_j, index=row, dim=0, dim_size=x.size(0))

        if self.concat:
            out = out.view(-1, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)

        out = out + self.bias

        if self.residual:
            if self.norm == 'neighbornorm':
                out = out + self.normlayer_r(self.lin_residual(x_in))
            else:
                out = out + self.lin_residual(x_in)

        if self.activation:
            out = F.elu(out, inplace=True)

        if self.norm == 'batchnorm':
            out = self.normlayer(out)
        elif self.norm == 'layernorm':
            out = self.normlayer(out)
        elif self.norm == 'pairnorm':
            out = self.normlayer(out)
        elif self.norm == 'nodenorm':
            out = self.normlayer(out)

        if self.dropout:
            out = F.dropout(out, p=0.5, training=self.training)

        return out
