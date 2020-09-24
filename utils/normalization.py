import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter, scatter_mean


class NodeNorm(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        return x


class PairNorm(torch.nn.Module):
    def __init__(self,
                 scale: float = 1.,
                 scale_individually: bool = False,
                 eps: float = 1e-5):
        super(PairNorm, self).__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        scale = self.scale

        if batch is None:
            x = x - x.mean(dim=0, keepdim=True)

            if not self.scale_individually:
                return scale * x / (self.eps + x.pow(2).sum(-1).mean()).sqrt()
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

        else:
            x = x - scatter(x, batch, dim=0, reduce='mean')[batch]

            if not self.scale_individually:
                return scale * x / torch.sqrt(
                    self.eps + scatter(x.pow(2).sum(-1, keepdim=True),
                                       batch,
                                       dim=0,
                                       reduce='mean')[batch])
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))


class SingleNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.Tensor(num_features),
                                        requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor(num_features),
                                       requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = (torch.var(x, dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        x = self.gamma * x + self.beta
        return x


class NeighborNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.Tensor(num_features),
                                        requires_grad=True)
        self.beta = torch.nn.Parameter(torch.Tensor(num_features),
                                       requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.gamma)
        torch.nn.init.zeros_(self.beta)

    def forward(self, x, edge_index):
        row, col = edge_index
        mean = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        mean = torch.mean(mean, dim=-1, keepdim=True)
        var = scatter_mean((x[col] - mean[row])**2,
                           row,
                           dim=0,
                           dim_size=x.size(0))
        var = torch.mean(var, dim=-1, keepdim=True)
        # std = scatter_std(x[col], row, dim=0, dim_size=x.size(0))
        out = (x[col] - mean[row]) / (var[row] + self.eps).sqrt()
        # out = (x[col] - mean[row]) / (std[row]**2 + self.eps).sqrt()
        out = self.gamma * out + self.beta
        return out


if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.utils import add_remaining_self_loops

    edge_index = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [2, 4], [3, 2], [4, 2]],
        dtype=torch.long)
    x = torch.tensor([[-1, 2, 3], [3, 2, 1], [1, 6, 9], [2, 3, 6], [3, 2, 8]],
                     dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous())
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    row, col = edge_index
    x = data.x
    print(x[col])
    neighbornorm = NeighborNorm(3)
    y = neighbornorm.forward(x, edge_index)
    print(y)
