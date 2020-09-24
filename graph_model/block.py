import torch
import torch.nn.functional as F
from utils.inits import glorot, zeros
from torch_geometric.nn import global_add_pool


class Block(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 out_dim,
                 num_layers=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out
        self.conv1 = model(in_dim=in_dim, out_dim=out_dim, norm='None')

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(model(in_dim=out_dim, out_dim=out_dim,
                                    norm=norm))

        self.lin = torch.nn.Linear(num_layers * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, batch)
        x = F.relu(self.lin(x))
        return x


class BlockHLA4(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 out_dim,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = Block(model=model,
                            in_dim=in_dim,
                            out_dim=out_dim,
                            num_layers=num_blocks,
                            norm=norm,
                            out=False)
        self.block2 = Block(model=model,
                            in_dim=out_dim,
                            out_dim=out_dim,
                            num_layers=num_blocks,
                            norm=norm,
                            out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, batch):
        x = self.block1(x, edge_index, batch)
        xs = [x]
        x = self.block2(x, edge_index, batch)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, batch)
        x = F.relu(self.lin(x))
        return x


class BlockHLA8(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 out_dim,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA4(model=model,
                                in_dim=in_dim,
                                out_dim=out_dim,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)
        self.block2 = BlockHLA4(model=model,
                                in_dim=out_dim,
                                out_dim=out_dim,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, batch):
        x = self.block1(x, edge_index, batch)
        xs = [x]

        x = self.block2(x, edge_index, batch)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, batch)
        x = F.relu(self.lin(x))
        return x


class BlockHLA16(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 out_dim,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA8(model=model,
                                in_dim=in_dim,
                                out_dim=out_dim,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)
        self.block2 = BlockHLA8(model=model,
                                in_dim=out_dim,
                                out_dim=out_dim,
                                num_blocks=num_blocks,
                                norm=norm,
                                out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, batch):
        x = self.block1(x, edge_index, batch)
        xs = [x]

        x = self.block2(x, edge_index, batch)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, batch)
        x = F.relu(self.lin(x))
        return x


class BlockHLA32(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 out_dim,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA16(model=model,
                                 in_dim=in_dim,
                                 out_dim=out_dim,
                                 num_blocks=num_blocks,
                                 norm=norm,
                                 out=False)
        self.block2 = BlockHLA16(model=model,
                                 in_dim=out_dim,
                                 out_dim=out_dim,
                                 num_blocks=num_blocks,
                                 norm=norm,
                                 out=False)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, batch):
        x = self.block1(x, edge_index, batch)
        xs = [x]

        x = self.block2(x, edge_index, batch)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, batch)
        x = F.relu(self.lin(x))
        return x


class BlockHLA64(torch.nn.Module):
    def __init__(self,
                 model,
                 in_dim,
                 out_dim,
                 num_blocks=2,
                 norm='neighbornorm',
                 out=False):
        super().__init__()
        self.out = out

        self.block1 = BlockHLA32(model=model,
                                 in_dim=in_dim,
                                 out_dim=out_dim,
                                 num_blocks=num_blocks,
                                 norm=norm)
        self.block2 = BlockHLA32(model=model,
                                 in_dim=out_dim,
                                 out_dim=out_dim,
                                 num_blocks=num_blocks,
                                 norm=norm)

        self.lin = torch.nn.Linear(num_blocks * out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.block1.reset_parameters()
        self.block2.reset_parameters()
        glorot(self.lin.weight)
        zeros(self.lin.bias)

    def forward(self, x, edge_index, batch):
        x = self.block1(x, edge_index, batch)
        xs = [x]

        x = self.block2(x, edge_index, batch)
        xs += [x]

        x = torch.cat(xs, dim=-1)
        if self.out:
            x = global_add_pool(x, batch)
        x = F.relu(self.lin(x))
        return x
