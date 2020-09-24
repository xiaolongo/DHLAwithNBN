import os.path as osp

import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def separate_data(dataset):
    labels = [g.y for g in dataset]
    idx_list = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    for fold_idx in range(10):
        train_idx, test_idx = idx_list[fold_idx]
        with open('train_idx-%d.txt' % (fold_idx + 1), 'w') as f:
            for index in train_idx:
                f.write(str(index))
                f.write('\n')
        with open('test_idx-%d.txt' % (fold_idx + 1), 'w') as f:
            for index in test_idx:
                f.write(str(index))
                f.write('\n')


def get_dataset(name, cleaned=True):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'graph_datasets', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset


if __name__ == '__main__':
    data = get_dataset('NCI1')
    print(data[0].num_features)
    # num_nodes = 0
    # for i in data:
    #     num_nodes += i.num_nodes
    # print(num_nodes)
    # separate_data(data)
