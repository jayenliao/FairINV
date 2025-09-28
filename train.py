import ipdb, time, random, os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm
from datetime import datetime
from args import get_args
from data import FairDataset
from utils import Results, set_seed, get_metrics
from models import ConstructModel, FairINV, EdgeAdder
from logger import EpochLogger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def make_cross_group_candidates(features: torch.Tensor,
                                sens: torch.Tensor,
                                A_base: SparseTensor,
                                k_per_node: int = 2,
                                device="cuda"):
    """
    Return LongTensor [2, M] of cross-group pairs with high cosine similarity,
    excluding already-connected pairs (according to A_base).
    """
    X = torch.nn.functional.normalize(features, p=2, dim=1).cpu()
    s = sens.cpu().long()
    idx0 = (s == 0).nonzero(as_tuple=True)[0]
    idx1 = (s == 1).nonzero(as_tuple=True)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Only one sensitive group present in data.")

    with torch.no_grad():
        S = X[idx0] @ X[idx1].T  # [|0|, |1|] cosine similarities
        k = min(k_per_node, S.shape[1])
        topv, topk = torch.topk(S, k=k, dim=1)
        I = idx0.repeat_interleave(k)          # sources in group 0
        J = idx1[topk.reshape(-1)]             # matched targets in group 1
        pairs = torch.stack([I, J], dim=0)     # [2, M]
        # Dedup (i,j) duplicates
        pairs = torch.unique(pairs, dim=1)

        # Remove already-connected pairs (check A_base)
        # A_base.has_value(i,j) is not exposed; so we query nonzero positions:
        base_row, base_col, _ = A_base.coo()
        base_set = set(zip(base_row.cpu().tolist(), base_col.cpu().tolist()))
        keep = []
        for a, b in pairs.T.tolist():
            if a != b and (a, b) not in base_set and (b, a) not in base_set:
                keep.append([a, b])
        if len(keep) == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        pairs = torch.tensor(keep, dtype=torch.long, device=device).T
    return pairs


def run_fairinv(args, data):
    torch.set_printoptions(threshold=float('inf'))
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
    elog = fairinv.train_model(data, pbar=args.pbar)

    """
    evaluation
    """
    fairinv.load_state_dict(torch.load(f'./weights/FairINV_{args.encoder}.pt'))
    fairinv.eval()
    with torch.no_grad():
        output = fairinv(data.features, data.edge_index)
    pred = (output.squeeze() > 0).type_as(data.labels)
    auc_test, f1_test, acc_test, dp_test, eo_test = get_metrics(
        Y=data.labels,
        logit=output,
        pred=pred,
        idx=data.idx_test,
        data=data
    )
    metrics_test = {
        'auc': auc_test,
        'f1': f1_test,
        'acc': acc_test,
        'dp': dp_test,
        'eo': eo_test
    }
    print("[TEST]", end=' ')
    for m, v in metrics_test.items():
        print(f"{m.upper():3}: {v:.4f}", end='  ')
    print()
    elog.log(args.epochs, "test", metrics_test)
    elog.close()

    return auc_test, f1_test, acc_test, dp_test, eo_test

def run(args, data, seed_dir):
    seed = int(seed_dir.split('seed_')[-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = data.features
    Y = data.labels
    EI = data.edge_index   # SparseTensor
    idx_tr, idx_va, idx_te = data.idx_train, data.idx_val, data.idx_test
    in_dim = X.shape[1]
    out_dim = 1
    backbone = ConstructModel(in_dim, args.hid_dim, args.encoder, args.layer_num).to(device)
    clf = nn.Linear(args.hid_dim, out_dim).to(device)

    use_edge_add = getattr(args, "model", "vanilla") == "edge_adder"
    # --- Edge adder module ---
    if use_edge_add:
        cand_ij = make_cross_group_candidates(
            X, data.sens, EI, k_per_node=getattr(args, "edge_k", 2), device=device)
        edge_adder = None
        if cand_ij.numel() > 0:
            edge_adder = EdgeAdder(X.shape[0], cand_ij, device=device).to(device)
            params = list(backbone.parameters()) + list(clf.parameters()) + list(edge_adder.parameters())
        else:
            params = list(backbone.parameters()) + list(clf.parameters())
            print("\033[31m[!] Warning: no cross-group candidate edges found; running vanilla training.\033[0m")
    else:
        edge_adder = None
        params = list(backbone.parameters()) + list(clf.parameters())

    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    pbar = tqdm(range(args.epochs), desc=f"{args.dataset}-{args.encoder}")
    best = {'score': -1e9, 'state': None}
    elog = EpochLogger(seed_dir)

    for ep in pbar:
        backbone.train()
        clf.train()
        opt.zero_grad()

        if edge_adder is not None:
            A_blend = (EI + edge_adder.sparse_tensor()).coalesce()
        else:
            A_blend = EI

        H = backbone(X, A_blend)          # [N, hid]
        logits = clf(H).squeeze(1)        # [N]
        probs  = torch.sigmoid(logits)
        bce = loss_fn(logits[idx_tr], Y[idx_tr].float())

        if edge_adder is not None:
            s_tr = data.sens[idx_tr].long()
            if (s_tr == 0).any() and (s_tr == 1).any():
                p0 = probs[idx_tr][s_tr == 0].mean()
                p1 = probs[idx_tr][s_tr == 1].mean()
                dp_loss = (p0 - p1).pow(2)
            else:
                raise ValueError("Only one sensitive group present in training data.")
            l1 = edge_adder.weights().sum() if edge_adder is not None else torch.zeros((), device=device)
            loss = bce + args.lambda_dp * dp_loss + args.lambda_edge_l1 * l1
        else:
            loss = bce
            dp_loss, l1 = None, None

        # --- Train metrics & logging ---
        with torch.no_grad():
            pred_tr = (logits > 0).long()
            auc_tr, f1_tr, acc_tr, dp_tr, eo_tr = get_metrics(
                Y, logits, pred=pred_tr, idx=idx_tr, data=data
            )
            metrics_train = {
                # losses (train)
                "loss": float(loss.item()),
                "bce": float(bce.item()),
                "dp_loss": float(dp_loss.item()) if dp_loss is not None else None,
                "l1": float(l1.item()) if l1 is not None else None,
                # metrics (train)
                "auc": auc_tr,
                "f1": f1_tr,
                "acc": acc_tr,
                "dp": dp_tr,
                "eo": eo_tr,
            }
            elog.log(ep, "train", metrics_train)

        loss.backward()
        opt.step()

        # --- Eval on val ---
        backbone.eval()
        clf.eval()
        with torch.no_grad():
            H_val = backbone(X, A_blend)
            logit_val = clf(H_val).squeeze(1)
            bce_val = loss_fn(logit_val[idx_va], Y[idx_va].float())
            probs_val = torch.sigmoid(logit_val)
            # DP loss on VAL (only when edge_adder is active)
            if edge_adder is not None:
                s_va = data.sens[idx_va].long()
                if (s_va == 0).any() and (s_va == 1).any():
                    pv0 = probs_val[idx_va][s_va == 0].mean()
                    pv1 = probs_val[idx_va][s_va == 1].mean()
                    dp_loss_val = (pv0 - pv1).pow(2)
                else:
                    dp_loss_val = torch.tensor(0.0, device=device)
                l1_val = l1  # same parameter regularizer; log under val for completeness
                loss_val_total = bce_val + args.lambda_dp * dp_loss_val + args.lambda_edge_l1 * l1_val
            else:
                dp_loss_val = None
                l1_val = None
                loss_val_total = bce_val

        pred_val = (logit_val > 0).long()
        auc, f1, acc, dp, eo = get_metrics(
            Y, logit_val, pred=pred_val, idx=idx_va, data=data
        )

        score = auc - dp - eo
        if score > best['score']:
            best['score'] = score
            best['state'] = {
                'backbone': backbone.state_dict(),
                'clf': clf.state_dict()
            }

        metrics_val = {
            # losses (val)
            'loss_val': float(loss_val_total.item()),
            'bce_val': float(bce_val.item()),
            'dp_loss_val': float(dp_loss_val.item()) if dp_loss_val is not None else None,
            'l1_val': float(l1_val.item()) if l1_val is not None and torch.is_tensor(l1_val) else (float(l1_val) if l1_val is not None else None),
            # metrics (val)
            'auc_val': auc,
            'f1_val': f1,
            'acc_val': acc,
            'dp_val': dp,
            'eo_val': eo
        }
        elog.log(ep, "val", metrics_val)

        if edge_adder is not None and dp_loss is not None and l1 is not None:
            message = f"loss: {loss.item():.3f}, bce: {bce.item():.3f}, dp_loss: {dp_loss.item():.3f}, l1: {l1.item():.3f}"
            message += f", auc: {auc:.3f}, f1: {f1:.3f}, acc: {acc:.3f}, dp: {dp:.3f}, eo: {eo:.3f}"
        else:
            message = f"loss(bce): {loss.item():.3f}"
            message += f", auc: {auc:.3f}, f1: {f1:.3f}, acc: {acc:.3f}, dp: {dp:.3f}, eo: {eo:.3f}"

        if (ep+1) % args.log_interval == 0:
            pbar.set_postfix({"Seed": seed, "Epoch": ep+1, "Message": message})

    # --- Test with the best checkpoint ---
    backbone.load_state_dict(best['state']['backbone'])
    clf.load_state_dict(best['state']['clf'])
    backbone.eval()
    clf.eval()
    with torch.no_grad():
        Ht = backbone(X, (EI + edge_adder.sparse_tensor()).coalesce() if edge_adder is not None else EI)
        logit_t = clf(Ht).squeeze(1)
    pred_t = (logit_t > 0).long()

    auc_test, f1_test, acc_test, dp_test, eo_test = get_metrics(
        Y, logit_t, pred=pred_t, idx=idx_te, data=data
    )
    metrics_test = {
        'auc_test': auc_test,
        'f1_test': f1_test,
        'acc_test': acc_test,
        'dp_test': dp_test,
        'eo_test': eo_test
    }
    elog.log(args.epochs, "test", metrics_test)
    elog.close()

    print(f"[TEST] AUC: {auc_test:.4f}  F1: {f1_test:.4f}  ACC: {acc_test:.4f}  DP: {dp_test:.4f}  EO: {eo_test:.4f}")

    return auc_test, f1_test, acc_test, dp_test, eo_test

def main(args):
    model_num = 1
    results = Results(args.seed_num, model_num, args)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f'{args.dataset}/{args.encoder}/{args.model}/{ts}'
    args.log_dir = os.path.join(args.log_dir, dir_name)

    data = FairDataset(args.dataset, args.device)
    data.load_data()
    data.info()

    for s in range(args.seed_num):
        seed = s + args.start_seed
        set_seed(seed)
        args.seed_dir = os.path.join(args.log_dir, f'seed_{seed}')
        os.makedirs(args.seed_dir, exist_ok=True)

        if args.model == "fairinv":
            args.pbar = tqdm(total=args.epochs, desc=f"Seed {seed}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
            auc, f1, acc, dp, eo = run_fairinv(args, data)
        elif args.model == "vanilla" or args.model == "edge_adder":
            auc, f1, acc, dp, eo = run(args, data, args.seed_dir)
        else:
            raise ValueError("Invalid mode. Choose 'fairinv' or 'vanilla'.")
        results.auc[s, :], results.f1[s, :], results.acc[s, :], \
            results.parity[s, :], results.equality[s, :] = auc, f1, acc, dp, eo

    results.report_results()
    if args.save_results:
        results.save_results(args)

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    main(args)
