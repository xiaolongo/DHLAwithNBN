import torch

from ppi_model.block import (Block, BlockHLA4, BlockHLA8, BlockHLA16,
                             BlockHLA32, BlockHLA64)
from ppi_model.layers import SAGELayer


class SAGENet(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 dropout=None,
                 norm=None,
                 residual=False,
                 mode=None):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.conv1 = SAGELayer(in_dim=in_dim,
                               out_dim=hidden_dim,
                               dropout=dropout,
                               norm=norm,
                               residual=residual,
                               activation=True)
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGELayer(in_dim=hidden_dim,
                          out_dim=hidden_dim,
                          dropout=dropout,
                          norm=norm,
                          residual=residual,
                          activation=True))

        self.convf = SAGELayer(in_dim=hidden_dim,
                               out_dim=out_dim,
                               dropout=False,
                               norm=norm,
                               residual=residual,
                               activation=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.convf(x, edge_index)
        return x


class SAGE_with_JK(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 dropout=False,
                 norm='None',
                 residual=False,
                 mode='mean'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        base_model = SAGELayer

        self.model = Block(base_model,
                           in_dim,
                           hidden_dim,
                           out_dim,
                           num_layers=num_layers,
                           dropout=dropout,
                           norm=norm,
                           residual=residual,
                           mode=self.mode)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x


class SAGE_with_DHLA(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 dropout=False,
                 norm='batchnorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        base_model = SAGELayer

        if num_layers == 2:
            self.model = Block(model=base_model,
                               in_dim=in_dim,
                               hidden_dim=hidden_dim,
                               out_dim=out_dim,
                               dropout=dropout,
                               norm=norm,
                               residual=residual,
                               mode=mode)

        elif num_layers == 4:
            self.model = BlockHLA4(model=base_model,
                                   in_dim=in_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=out_dim,
                                   dropout=dropout,
                                   norm=norm,
                                   residual=residual,
                                   mode=mode)

        elif num_layers == 8:
            self.model = BlockHLA8(model=base_model,
                                   in_dim=in_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=out_dim,
                                   dropout=dropout,
                                   norm=norm,
                                   residual=residual,
                                   mode=mode)

        elif num_layers == 16:
            self.model = BlockHLA16(model=base_model,
                                    in_dim=in_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=out_dim,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)

        elif num_layers == 32:
            self.model = BlockHLA32(model=base_model,
                                    in_dim=in_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=out_dim,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)

        else:
            self.model = BlockHLA64(model=base_model,
                                    in_dim=in_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=out_dim,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x
