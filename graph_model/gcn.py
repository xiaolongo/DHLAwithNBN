import torch

from graph_model.block import (Block, BlockHLA4, BlockHLA8, BlockHLA16,
                               BlockHLA32, BlockHLA64)
from graph_model.layers import GCNLayer
from utils.inits import glorot, zeros


class GCNNet(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, norm='None'):
        super().__init__()
        in_dim = dataset.num_features
        out_dim = dataset.num_classes
        model = GCNLayer
        self.block = Block(model,
                           in_dim=in_dim,
                           out_dim=hidden,
                           num_layers=num_layers,
                           norm=norm, 
                           out=True)
        self.lin = torch.nn.Linear(hidden, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.block.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_g = self.block(x, edge_index, batch)
        out = self.lin(x_g)
        return out.log_softmax(dim=-1)


class GCNDHLA(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, norm='None'):
        super().__init__()
        in_dim = dataset.num_features
        out_dim = dataset.num_classes
        model = GCNLayer
        if num_layers == 2:
            self.block = Block(model,
                               in_dim=in_dim,
                               out_dim=hidden,
                               norm=norm,
                               out=True)
        elif num_layers == 4:
            self.block = BlockHLA4(model,
                                   in_dim=in_dim,
                                   out_dim=hidden,
                                   norm=norm,
                                   out=True)
        elif num_layers == 8:
            self.block = BlockHLA8(model,
                                   in_dim=in_dim,
                                   out_dim=hidden,
                                   norm=norm,
                                   out=True)
        elif num_layers == 16:
            self.block = BlockHLA16(model,
                                    in_dim=in_dim,
                                    out_dim=hidden,
                                    norm=norm,
                                    out=True)
        elif num_layers == 32:
            self.block = BlockHLA32(model,
                                    in_dim=in_dim,
                                    out_dim=hidden,
                                    norm=norm,
                                    out=True)
        elif num_layers == 64:
            self.block = BlockHLA64(model,
                                    in_dim=in_dim,
                                    out_dim=hidden,
                                    norm=norm,
                                    out=True)
        self.lin = torch.nn.Linear(hidden, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.block.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_g = self.block(x, edge_index, batch)
        out = self.lin(x_g)
        return out.log_softmax(dim=-1)
