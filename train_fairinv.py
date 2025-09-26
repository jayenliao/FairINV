# %%
# import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import warnings
from torch_geometric.loader import DataLoader
from datetime import datetime

warnings.filterwarnings('ignore')

from load_data import *
# from models import *
from utils import set_seed
import torch.nn as nn
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from fairinv import *
from logger import EpochLogger
import json


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fairinv','vanilla'], default='vanilla')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--start_seed', type=int, default=42, help='Random seed start.')
    parser.add_argument('--seed_num', type=int, default=0, help='The number of random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hid_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'german'])
    parser.add_argument("--layer_num", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn','gat','gin','sage','sgc'])
    parser.add_argument('--aggr', type=str, default='add',
                        choices=['add', 'mean', 'max', 'min', 'sum', 'std', 'var', 'median'],
                        help="aggregation function")
    parser.add_argument('--weight_path', type=str, default='./weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='hyperpapameter to balance the downstream task and invariance learning loss.')
    parser.add_argument('--lr_sp', type=float, default=0.5, help='the learning rate of the sensitive partition.')
    parser.add_argument('--env_num', type=int, default=2,
                        help='the number of the sensitive attribute, also known as environment number.')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--log_interval', type=int, default=20, help='Interval for logging.')
    parser.add_argument('--partition_times', type=int, default=3,
                        help='the number for partitioning the sensitive attribute group.')

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_fairinv(args):
    torch.set_printoptions(threshold=float('inf'))
    """
    Load data
    """
    data = FairDataset(args.dataset, args.device)
    data.load_data()

    num_class = 1
    args.in_dim = data.features.shape[1]
    args.nnode = data.features.shape[0]
    args.out_dim = num_class

    """
    Build model, optimizer, and loss fuction
    """
    # FairINV
    fairinv = FairINV(args)

    """
    Train model
    """
    fairinv.train_model(data, pbar=args.pbar)

    """
    evaluation
    """
    fairinv.load_state_dict(torch.load(f'./weights/FairINV_{args.encoder}.pt'))
    fairinv.eval()
    with torch.no_grad():
        output = fairinv(data.features, data.edge_index)

    pred = (output.squeeze() > 0).type_as(data.labels)
    # utility performance
    auc_test = roc_auc_score(data.labels[data.idx_test].cpu(), output[data.idx_test].cpu())
    f1_test = f1_score(data.labels[data.idx_test].cpu(), pred[data.idx_test].cpu())
    acc_test = accuracy_score(data.labels[data.idx_test].cpu(), pred[data.idx_test].cpu())
    # fairness performance
    parity_test, equality_test = fair_metric(pred[data.idx_test].cpu().numpy(),
                                             data.labels[data.idx_test].cpu().numpy(),
                                             data.sens[data.idx_test].cpu().numpy())

    return auc_test, f1_test, acc_test, parity_test, equality_test

def run_vanilla(args):
    args = args_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load data (outputs PyG SparseTensor edge_index as data.edge_index) ---
    data = FairDataset(args.dataset, device)
    data.load_data()  # uses your loaders and sets .features/.labels/.sens/.idx_*
    X = data.features
    Y = data.labels
    EI = data.edge_index   # SparseTensor
    idx_tr, idx_va, idx_te = data.idx_train, data.idx_val, data.idx_test

    in_dim = X.shape[1]; out_dim = 1

    # --- Vanilla backbone + linear classifier ---
    backbone = ConstructModel(in_dim, args.hid_dim, args.encoder, args.layer_num).to(device)
    clf = nn.Linear(args.hid_dim, out_dim).to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(clf.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    pbar = tqdm(range(args.epochs), desc=f"{args.dataset}-{args.encoder}")
    best = {'score': -1e9, 'state': None}
    elog = EpochLogger(args.seed_dir)
    for ep in pbar:
        backbone.train(); clf.train()
        opt.zero_grad()
        H = backbone(X, EI)          # [N, hid]
        logits = clf(H).squeeze(1)   # [N]
        loss = loss_fn(logits[idx_tr], Y[idx_tr].float())
        loss.backward(); opt.step()

        # --- Eval on val ---
        backbone.eval(); clf.eval()
        with torch.no_grad():
            H_val = backbone(X, EI)
            logit_val = clf(H_val).squeeze(1)
            loss_val = loss_fn(logit_val[idx_va], Y[idx_va].float()).item()
        pred_val = (logit_val > 0).long()

        auc = roc_auc_score(Y[idx_va].cpu(), logit_val[idx_va].cpu())
        f1  = f1_score(Y[idx_va].cpu(), pred_val[idx_va].cpu())
        acc = accuracy_score(Y[idx_va].cpu(), pred_val[idx_va].cpu())
        dp, eo = fair_metric(pred_val[idx_va].cpu().numpy(),
                             Y[idx_va].cpu().numpy(),
                             data.sens[idx_va].cpu().numpy())

        score = auc - dp - eo
        if score > best['score']:
            best['score'] = score
            best['state'] = {
                'backbone': backbone.state_dict(),
                'clf': clf.state_dict()
            }

        metrics_val = {
            'loss_val': loss_val,
            'auc_val': auc,
            'f1_val': f1,
            'acc_val': acc,
            'dp_val': dp,
            'eo_val': eo
        }
        elog.log(ep, "val", metrics_val)

        if (ep+1) % args.log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}", auc=f"{auc:.3f}", f1=f"{f1:.3f}", acc=f"{acc:.3f}", dp=f"{dp:.3f}", eo=f"{eo:.3f}")


    # --- Test with the best checkpoint ---
    backbone.load_state_dict(best['state']['backbone']); clf.load_state_dict(best['state']['clf'])
    backbone.eval(); clf.eval()
    with torch.no_grad():
        Ht = backbone(X, EI)
        logit_t = clf(Ht).squeeze(1)
    pred_t = (logit_t > 0).long()

    auc_t = roc_auc_score(Y[idx_te].cpu(), logit_t[idx_te].cpu())
    f1_t  = f1_score(Y[idx_te].cpu(), pred_t[idx_te].cpu())
    acc_t = accuracy_score(Y[idx_te].cpu(), pred_t[idx_te].cpu())
    dp_t, eo_t = fair_metric(pred_t[idx_te].cpu().numpy(),
                             Y[idx_te].cpu().numpy(),
                             data.sens[idx_te].cpu().numpy())
    metrics_test = {
        'auc_test': auc_t,
        'f1_test': f1_t,
        'acc_test': acc_t,
        'dp_test': dp_t,
        'eo_test': eo_t
    }
    elog.log(args.epochs, "test", metrics_test)

    print(f"[TEST] AUC: {auc_t:.4f}  F1: {f1_t:.4f}  ACC: {acc_t:.4f}  DP: {dp_t:.4f}  EO: {eo_t:.4f}")

    return auc_t, f1_t, acc_t, dp_t, eo_t

def main(args):
    model_num = 1
    results = Results(args.seed_num, model_num, args)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f'{args.model}_{args.dataset}_{ts}'
    args.log_dir = os.path.join(args.log_dir, dir_name)

    for s in range(args.seed_num):
        seed = s + args.start_seed
        set_seed(seed)
        args.seed_dir = os.path.join(args.log_dir, f'seed_{seed}')
        os.makedirs(args.seed_dir, exist_ok=True)

        print(f"Seed={seed}")
        if args.model == "fairinv":
            args.pbar = tqdm(total=args.epochs, desc=f"Seed {seed + 1}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
            auc, f1, acc, dp, eo = run_fairinv(args)
        elif args.model == "vanilla":
            auc, f1, acc, dp, eo = run_vanilla(args)
        else:
            raise ValueError("Invalid mode. Choose 'fairinv' or 'vanilla'.")
        results.auc[s, :], results.f1[s, :], results.acc[s, :], \
            results.parity[s, :], results.equality[s, :] = auc, f1, acc, dp, eo

    results.report_results()
    if args.save_results:
        results.save_results(args)

if __name__ == '__main__':
    args = args_parser()
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    main(args)
