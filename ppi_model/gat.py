import torch

from ppi_model.block import (GATBlock, GATBlockHLA4, GATBlockHLA8,
                             GATBlockHLA16, GATBlockHLA32, GATBlockHLA64)
from ppi_model.layers import GATLayer


class GATNet(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 heads=4,
                 dropout=False,
                 norm=None,
                 residual=False,
                 mode=None):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.conv1 = GATLayer(in_dim=in_dim,
                              out_dim=hidden_dim,
                              dropout=dropout,
                              heads=heads,
                              norm=norm,
                              residual=residual,
                              activation=True)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATLayer(in_dim=heads * hidden_dim,
                         out_dim=hidden_dim,
                         dropout=dropout,
                         norm=norm,
                         residual=residual,
                         activation=True))

        self.convf = GATLayer(in_dim=heads * hidden_dim,
                              out_dim=out_dim,
                              heads=6,
                              dropout=False,
                              concat=False,
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


class GAT_with_JK(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 heads=4,
                 dropout=False,
                 norm='None',
                 residual=False,
                 mode='mean'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.model = GATBlock(GATLayer,
                              in_dim,
                              hidden_dim,
                              out_dim,
                              heads=heads,
                              dropout=dropout,
                              num_layers=num_layers,
                              norm=norm,
                              residual=residual,
                              mode=self.mode)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x


class GAT_with_DHLA(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 heads=4,
                 dropout=False,
                 norm='batchnorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']
        assert num_layers in [2, 4, 8, 16, 32, 64]

        base_model = GATLayer

        if num_layers == 2:
            self.model = GATBlock(model=base_model,
                                  in_dim=in_dim,
                                  hidden_dim=hidden_dim,
                                  out_dim=out_dim,
                                  heads=heads,
                                  dropout=dropout,
                                  norm=norm,
                                  residual=residual,
                                  mode=mode)

        elif num_layers == 4:
            self.model = GATBlockHLA4(model=base_model,
                                      in_dim=in_dim,
                                      hidden_dim=hidden_dim,
                                      out_dim=out_dim,
                                      heads=heads,
                                      norm=norm,
                                      residual=residual,
                                      mode=mode)

        elif num_layers == 8:
            self.model = GATBlockHLA8(model=base_model,
                                      in_dim=in_dim,
                                      hidden_dim=hidden_dim,
                                      out_dim=out_dim,
                                      heads=heads,
                                      norm=norm,
                                      residual=residual,
                                      mode=mode)

        elif num_layers == 16:
            self.model = GATBlockHLA16(model=base_model,
                                       in_dim=in_dim,
                                       hidden_dim=hidden_dim,
                                       out_dim=out_dim,
                                       heads=heads,
                                       norm=norm,
                                       residual=residual,
                                       mode=mode)

        elif num_layers == 32:
            self.model = GATBlockHLA32(model=base_model,
                                       in_dim=in_dim,
                                       hidden_dim=hidden_dim,
                                       out_dim=out_dim,
                                       heads=heads,
                                       norm=norm,
                                       residual=residual,
                                       mode=mode)

        else:
            self.model = GATBlockHLA64(model=base_model,
                                       in_dim=in_dim,
                                       hidden_dim=hidden_dim,
                                       out_dim=out_dim,
                                       heads=heads,
                                       norm=norm,
                                       residual=residual,
                                       mode=mode)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x
