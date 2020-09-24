import os.path as osp
import torch

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (PPI, Amazon, GNNBenchmarkDataset,
                                      Planetoid, Reddit)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def get_planetoid_dataset(name, normalize_features=True, transform=None):
    '''
        Cora: train => 232*7; val => 542; test => 542;
        CiteSeer: train => 333*6; val => 664; test => 665;
        PubMed: train => 3943*3; val => 3944; test => 3944;
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'node_datasets')
    dataset = Planetoid(path, name)
    # if name == 'Cora':
    #     dataset = Planetoid(path,
    #                         name,
    #                         split='random',
    #                         num_train_per_class=232,
    #                         num_val=542,
    #                         num_test=542)
    # elif name == 'CiteSeer':
    #     dataset = Planetoid(path,
    #                         name,
    #                         split='random',
    #                         num_train_per_class=333,
    #                         num_val=664,
    #                         num_test=665)
    # else:
    #     dataset = Planetoid(path,
    #                         name,
    #                         split='random',
    #                         num_train_per_class=3943,
    #                         num_val=3944,
    #                         num_test=3944)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset


def get_ppi_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'node_datasets', name)
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    dataset = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    return dataset


def get_gbd_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'node_datasets')
    train_dataset = GNNBenchmarkDataset(path, name, split='train')
    val_dataset = GNNBenchmarkDataset(path, name, split='val')
    test_dataset = GNNBenchmarkDataset(path, name, split='test')

    dataset = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    return dataset


def get_reddit_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'node_data', name)
    dataset = Reddit(path)
    return dataset


def get_amazon_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'node_datasets', name)
    dataset = Amazon(path, name, transform=T.NormalizeFeatures())

    num_per_class = 20
    train_index = []
    val_index = []
    test_index = []
    for i in range(dataset.num_classes):
        index = (dataset[0].y.long() == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        if len(index) > num_per_class + 30:
            train_index.append(index[:num_per_class])
            val_index.append(index[num_per_class:num_per_class + 30])
            test_index.append(index[num_per_class + 30:])
        else:
            continue
    train_index = torch.cat(train_index)
    val_index = torch.cat(val_index)
    test_index = torch.cat(test_index)

    train_mask = index_to_mask(train_index, size=dataset[0].num_nodes)
    val_mask = index_to_mask(val_index, size=dataset[0].num_nodes)
    test_mask = index_to_mask(test_index, size=dataset[0].num_nodes)

    dataset.train_mask = train_mask
    dataset.val_mask = val_mask
    dataset.test_mask = test_mask

    return dataset


def get_ogb_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets',
                    'node_datasets', name)
    dataset = PygNodePropPredDataset(name, root=path)
    return dataset


def get_dataset(name, normalize_features=False, transform=None):
    planetoid = ['Cora', 'CiteSeer', 'PubMed']
    gbd = ['PATTERN', 'CLUSTER']
    reddit = ['Reddit']
    ogb = ['ogbn-arxiv']
    amazon = ['Computers', 'Photo']

    if name in planetoid:
        dataset = get_planetoid_dataset(name, normalize_features, transform)
    elif name in gbd:
        dataset = get_gbd_dataset(name)
    elif name in reddit:
        dataset = get_reddit_dataset(name)
    elif name in amazon:
        dataset = get_amazon_dataset(name)
    elif name in ogb:
        dataset = get_ogb_dataset(name)
    else:
        dataset = get_ppi_dataset(name)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset('Computers')
    data = dataset[0]
