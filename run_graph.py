import argparse
import os.path as osp
import setproctitle
import torch
import numpy as np
from torch import tensor
from torch_geometric.data import DataLoader

from graph_model.gin import GINNet, GINDHLA
from graph_model.gcn import GCNNet, GCNDHLA
from graph_model.sage import SAGENet, SAGEDHLA
from graph_model.train_eval import eval_acc, eval_loss, train
from utils.early_stopping import EarlyStopping
from utils.get_graph_data import get_dataset

setproctitle.setproctitle('fxl')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='SAGEDHLA')
parser.add_argument('--dataset', type=str, default='NCI1')
parser.add_argument('--num_layers', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--norm', type=str, default='neighbornorm')
parser.add_argument('--logger', action='store_true')
args = parser.parse_args()

device = torch.device(f'cuda: {args.gpu_id}')
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

dataset = get_dataset(name=args.dataset)
accs = []
for idx in range(1, 10 + 1):
    index_train = []
    index_test = []
    with open(
            osp.join(osp.dirname(osp.realpath(__file__)), 'datasets',
                     'graph_datasets', '%s' % args.dataset, '10fold_idx',
                     'train_idx-%d.txt' % idx), 'r') as f_train:
        for line in f_train:
            index_train.append(int(line.split('\n')[0]))
    with open(
            osp.join(osp.dirname(osp.realpath(__file__)), 'datasets',
                     'graph_datasets', '%s' % args.dataset, '10fold_idx',
                     'test_idx-%d.txt' % idx), 'r') as f_test:
        for line in f_test:
            index_test.append(int(line.split('\n')[0]))

    train_dataset = [dataset[i] for i in index_train]
    test_dataset = [dataset[j] for j in index_test]
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    models = {
        'GIN': GINNet,
        'GINDHLA': GINDHLA,
        'GCN': GCNNet,
        'GCNDHLA': GCNDHLA,
        'SAGE': SAGENet,
        'SAGEDHLA': SAGEDHLA
    }
    assert args.model in models
    model = models[args.model]
    model = model(dataset, args.num_layers, args.hidden, args.norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(patience=args.patience)
    train_loss, test_acc = 0, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optimizer, train_loader, device)
        test_loss = eval_loss(model, test_loader, device)
        test_acc = eval_acc(model, test_loader, device)
        if args.logger:
            print(f'{idx}/{epoch} '
                  f'Train Loss:{train_loss:.4f} '
                  f'Test Loss:{test_loss:.4f} '
                  f'Test Acc:{test_acc:.4f}')
        early_stopping(test_loss, test_acc)
        if early_stopping.early_stop:
            best_val_loss = early_stopping.best_score
            best_test_acc = early_stopping.best_score_acc
            print(f'Best Test Loss: {best_val_loss:.3f} '
                  f'Best Test Acc: {best_test_acc:.4f}')
            accs.append(best_test_acc)
            break

acc = tensor(accs)
acc_mean = acc.mean().item()
acc_std = acc.std().item()
print(f'Final Test Accuracy: {acc_mean:.4f} ± {acc_std:.4f}')
