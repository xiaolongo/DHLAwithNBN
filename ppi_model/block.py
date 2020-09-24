import torch
import torch.nn.functional as F
from utils.inits import glorot, zeros


class Block(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.conv1 = model(in_dim=in_dim,
                           out_dim=hidden_dim,
                           dropout=dropout,
                           norm=norm,
                           residual=residual,
                           activation=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                model(in_dim=hidden_dim,
                      out_dim=hidden_dim,
                      dropout=dropout,
                      norm=norm,
                      residual=residual,
                      activation=True))

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_layers, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        xs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class BlockHLA4(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = Block(model=model,
                            in_dim=in_dim,
                            hidden_dim=hidden_dim,
                            out_dim=hidden_dim,
                            num_layers=num_blocks,
                            dropout=dropout,
                            norm=norm,
                            residual=residual,
                            mode=mode)
        self.block2 = Block(model=model,
                            in_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            out_dim=hidden_dim,
                            num_layers=num_blocks,
                            dropout=dropout,
                            norm=norm,
                            residual=residual,
                            mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index))
        xs = [x]

        x = F.relu(self.block2(x, edge_index))
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class BlockHLA8(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = BlockHLA4(model=model,
                                in_dim=in_dim,
                                hidden_dim=hidden_dim,
                                out_dim=hidden_dim,
                                num_blocks=num_blocks,
                                dropout=dropout,
                                norm=norm,
                                residual=residual,
                                mode=mode)
        self.block2 = BlockHLA4(model=model,
                                in_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                out_dim=hidden_dim,
                                num_blocks=num_blocks,
                                dropout=dropout,
                                norm=norm,
                                residual=residual,
                                mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index))
        xs = [x]

        x = self.block2(x, edge_index)
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class BlockHLA16(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = BlockHLA8(model=model,
                                in_dim=in_dim,
                                hidden_dim=hidden_dim,
                                out_dim=hidden_dim,
                                num_blocks=num_blocks,
                                dropout=dropout,
                                norm=norm,
                                residual=residual,
                                mode=mode)
        self.block2 = BlockHLA8(model=model,
                                in_dim=hidden_dim,
                                hidden_dim=hidden_dim,
                                out_dim=hidden_dim,
                                num_blocks=num_blocks,
                                dropout=dropout,
                                norm=norm,
                                residual=residual,
                                mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index))
        xs = [x]

        x = F.relu(self.block2(x, edge_index))
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class BlockHLA32(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = BlockHLA16(model=model,
                                 in_dim=in_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_blocks=num_blocks,
                                 dropout=dropout,
                                 norm=norm,
                                 residual=residual,
                                 mode=mode)
        self.block2 = BlockHLA16(model=model,
                                 in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_blocks=num_blocks,
                                 dropout=dropout,
                                 norm=norm,
                                 residual=residual,
                                 mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index))
        xs = [x]

        x = F.relu(self.block2(x, edge_index))
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class BlockHLA64(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = BlockHLA32(model=model,
                                 in_dim=in_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_blocks=num_blocks,
                                 dropout=dropout,
                                 norm=norm,
                                 residual=residual,
                                 mode=mode)
        self.block2 = BlockHLA32(model=model,
                                 in_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=hidden_dim,
                                 num_blocks=num_blocks,
                                 dropout=dropout,
                                 norm=norm,
                                 residual=residual,
                                 mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index))
        xs = [x]

        x = F.relu(self.block2(x, edge_index))
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class GATBlock(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads=4,
                 num_layers=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.conv1 = model(in_dim=in_dim,
                           out_dim=hidden_dim,
                           heads=heads,
                           dropout=dropout,
                           norm=norm,
                           residual=residual,
                           activation=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                model(in_dim=heads * hidden_dim,
                      out_dim=hidden_dim,
                      heads=heads,
                      dropout=dropout,
                      norm=norm,
                      residual=residual,
                      activation=True))

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(heads * hidden_dim * num_layers,
                                       out_dim)
        else:
            self.lin = torch.nn.Linear(heads * hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        xs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class GATBlockHLA4(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads=4,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = GATBlock(model=model,
                               in_dim=in_dim,
                               hidden_dim=hidden_dim,
                               out_dim=hidden_dim,
                               heads=heads,
                               num_layers=num_blocks,
                               dropout=dropout,
                               norm=norm,
                               residual=residual,
                               mode=mode)
        self.block2 = GATBlock(model=model,
                               in_dim=hidden_dim,
                               hidden_dim=hidden_dim,
                               out_dim=hidden_dim,
                               heads=heads,
                               num_layers=num_blocks,
                               dropout=dropout,
                               norm=norm,
                               residual=residual,
                               mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index), inplace=True)
        xs = [x]

        x = F.relu(self.block2(x, edge_index), inplace=True)
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class GATBlockHLA8(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads=4,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = GATBlockHLA4(model=model,
                                   in_dim=in_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=hidden_dim,
                                   heads=heads,
                                   num_blocks=num_blocks,
                                   dropout=dropout,
                                   norm=norm,
                                   residual=residual,
                                   mode=mode)
        self.block2 = GATBlockHLA4(model=model,
                                   in_dim=hidden_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=hidden_dim,
                                   heads=heads,
                                   num_blocks=num_blocks,
                                   dropout=dropout,
                                   norm=norm,
                                   residual=residual,
                                   mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index), inplace=True)
        xs = [x]

        x = F.relu(self.block2(x, edge_index), inplace=True)
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class GATBlockHLA16(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads=4,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = GATBlockHLA8(model=model,
                                   in_dim=in_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=hidden_dim,
                                   heads=heads,
                                   num_blocks=num_blocks,
                                   dropout=dropout,
                                   norm=norm,
                                   residual=residual,
                                   mode=mode)
        self.block2 = GATBlockHLA8(model=model,
                                   in_dim=hidden_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=hidden_dim,
                                   heads=heads,
                                   num_blocks=num_blocks,
                                   dropout=dropout,
                                   norm=norm,
                                   residual=residual,
                                   mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index), inplace=True)
        xs = [x]

        x = F.relu(self.block2(x, edge_index), inplace=True)
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class GATBlockHLA32(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads=4,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = GATBlockHLA16(model=model,
                                    in_dim=in_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=hidden_dim,
                                    heads=heads,
                                    num_blocks=num_blocks,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)
        self.block2 = GATBlockHLA16(model=model,
                                    in_dim=hidden_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=hidden_dim,
                                    heads=heads,
                                    num_blocks=num_blocks,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index), inplace=True)
        xs = [x]

        x = F.relu(self.block2(x, edge_index), inplace=True)
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x


class GATBlockHLA64(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 heads=4,
                 num_blocks=2,
                 dropout=False,
                 norm='neighbornorm',
                 residual=False,
                 mode='sum'):
        super().__init__()
        self.mode = mode
        assert self.mode in ['sum', 'mean', 'max', 'concat']

        self.block1 = GATBlockHLA32(model=model,
                                    in_dim=in_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=hidden_dim,
                                    heads=heads,
                                    num_blocks=num_blocks,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)
        self.block2 = GATBlockHLA32(model=model,
                                    in_dim=hidden_dim,
                                    hidden_dim=hidden_dim,
                                    out_dim=hidden_dim,
                                    heads=heads,
                                    num_blocks=num_blocks,
                                    dropout=dropout,
                                    norm=norm,
                                    residual=residual,
                                    mode=mode)

        if self.mode == 'concat':
            self.lin = torch.nn.Linear(hidden_dim * num_blocks, out_dim)
        else:
            self.lin = torch.nn.Linear(hidden_dim, out_dim)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index):
        x = F.relu(self.block1(x, edge_index), inplace=True)
        xs = [x]

        x = F.relu(self.block2(x, edge_index), inplace=True)
        xs += [x]

        if self.mode == 'sum':
            x = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            x = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'max':  # max
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'concat':
            x = torch.cat(xs, dim=-1)

        x = self.lin(x)
        return x
