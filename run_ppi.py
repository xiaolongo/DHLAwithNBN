# import os.path as osp

import argparse

import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch_geometric.data import DataLoader

from ppi_model.gat import GAT_with_DHLA, GAT_with_JK, GATNet
from ppi_model.gcn import GCN_with_DHLA, GCN_with_JK, GCNNet
from ppi_model.sage import SAGE_with_DHLA, SAGE_with_JK, SAGENet
from utils.early_stopping import EarlyStopping
from utils.get_node_data import get_dataset

# from torch.utils.tensorboard import SummaryWriter


def train_ppi(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy_with_logits(out, data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def val_ppi(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            num_graphs = data.num_graphs
            data = data.to(device)
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            total_loss += loss.item() * num_graphs
    return total_loss / len(val_loader.dataset)


def test_ppi(model, test_loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in test_loader:
            ys.append(data.y)
            out = model(data.to(device))
            preds.append((out > 0).float().cpu())
        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def run_ppi(train_loader,
            val_loader,
            test_loader,
            model,
            epochs,
            lr,
            weight_decay,
            patience,
            device,
            logger=True):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode='min',
    #                                                        factor=0.5,
    #                                                        patience=20)
    early_stopping = EarlyStopping(patience=patience)

    # path1 = osp.join(osp.dirname(osp.realpath(__file__)), 'runs', 'train')
    # path2 = osp.join(osp.dirname(osp.realpath(__file__)), 'runs', 'val')

    # writer_train = SummaryWriter(path1)
    # writer_val = SummaryWriter(path2)

    for epoch in range(1, epochs + 1):

        train_loss = train_ppi(model, optimizer, train_loader, device)

        val_loss = val_ppi(model, val_loader, device)
        test_f1 = test_ppi(model, test_loader, device)
        # scheduler.step(val_loss)

        # writer_train.add_scalar('training loss', train_loss, epoch)
        # writer_val.add_scalar('val loss', val_loss, epoch)

        if logger:
            print(
                '{:03d}: Train Loss: {:.4f}, Val Loss: {:.4f}, Test F1: {:.4f}'
                .format(epoch, train_loss, val_loss, test_f1))

        early_stopping(val_loss, test_f1)
        if early_stopping.early_stop:
            best_val_loss = early_stopping.best_score
            best_test_f1 = early_stopping.best_score_acc
            print('Val Loss: {:.3f}, Test F1 Score: {:.4f}'.format(
                best_val_loss, best_test_f1))
            break


if __name__ == '__main__':
    setproctitle.setproctitle('fxl')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--mode', type=str, default='mean')
    parser.add_argument('--dataset', type=str, default='PPI')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--norm', type=str, default='neighbornorm')
    parser.add_argument('--residual', type=bool, default=True)    
    parser.add_argument('--logger', action='store_false')
    args = parser.parse_args()

    device = torch.device(f'cuda: {args.gpu_id}')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    models = {
        'GCN': GCNNet,
        'GCN_with_JK': GCN_with_JK,
        'GCN_with_DHLA': GCN_with_DHLA,
        'SAGE': SAGENet,
        'SAGE_with_JK': SAGE_with_JK,
        'SAGE_with_DHLA': SAGE_with_DHLA,
        'GAT': GATNet,
        'GAT_with_JK': GAT_with_JK,
        'GAT_with_DHLA': GAT_with_DHLA
    }
    assert args.model in models
    model = models[args.model]

    dataset = get_dataset(args.dataset)
    train_loader = DataLoader(dataset['train'], batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset['val'], batch_size=2, shuffle=False)
    test_loader = DataLoader(dataset['test'], batch_size=2, shuffle=False)
    model = model(in_dim=dataset['train'].num_features,
                  hidden_dim=args.hidden,
                  out_dim=dataset['train'].num_classes,
                  num_layers=args.num_layers,
                  norm=args.norm,
                  residual=args.residual,
                  mode=args.mode)

    run_ppi(train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
            logger=args.logger)
