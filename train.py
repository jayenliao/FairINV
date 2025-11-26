import ipdb, time, random, os, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from tqdm import tqdm
from datetime import datetime
from args import get_parser
from data import FairDataset
from utils import Results, configure_threads, set_seed, get_metrics
from policies import build_policies
from models import ConstructModel, FairINV, EdgeAdder
from logger import EpochLogger
from nifa_bridge import apply_nifa_attack
from torch_sparse import SparseTensor
from types import SimpleNamespace

def snapshot_clean_data(data):
    """Lightweight immutable snapshot on CPU; safe for per-seed restores."""
    snap = {
        "features":   data.features.detach().cpu().clone(),
        "labels":     data.labels.detach().cpu().clone(),
        "sens":       data.sens.detach().cpu().clone(),
        "edge_index": data.edge_index_nor.detach().cpu().clone(),  # (2,E)
        "idx_train":  data.idx_train.detach().cpu().clone(),
        "idx_val":    data.idx_val.detach().cpu().clone(),
        "idx_test":   data.idx_test.detach().cpu().clone(),
    }
    return snap

def restore_from_snapshot(snap, device):
    """New data object with tensors moved to device and SparseTensor rebuilt."""
    data = SimpleNamespace()
    data.features   = snap["features"].to(device)
    data.labels     = snap["labels"].to(device).long()
    data.sens       = snap["sens"].to(device).float()
    data.edge_index_nor = snap["edge_index"].to(device).long()
    N = int(data.features.size(0))
    data.edge_index = SparseTensor.from_edge_index(
        data.edge_index_nor, sparse_sizes=(N, N)
    ).coalesce()
    data.idx_train  = snap["idx_train"].to(device).long()
    data.idx_val    = snap["idx_val"].to(device).long()
    data.idx_test   = snap["idx_test"].to(device).long()
    return data


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
        topv, topk = torch.topk(S, k=k, dim=1) # [N, k]
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


def run_fairinv(args, data, pbar):
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
    checkpoint_path = f'./weights/fi/{args.dataset}_{args.encoder}.pt' if not args.debug else f'./weights/fi_old/FairINV_{args.encoder}_debug.pt'
    elog = fairinv.train_model(data, checkpoint_path, pbar=pbar)

    """
    evaluation
    """
    fairinv.load_state_dict(torch.load(checkpoint_path))
    fairinv.eval()
    with torch.no_grad():
        output = fairinv(data.features, data.edge_index)
    pred = (output.squeeze() > 0).type_as(data.labels)
    auc_test, f1_test, acc_test, dp_test, eo_test = get_metrics(
        Y=data.labels,
        logit=output,
        pred=pred,
        idx=data.idx_test,
        data=data,
        neg=False
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

def _soft_dp_from_logits(logits: torch.Tensor, sens: torch.Tensor, idx: torch.Tensor):
    """Soft demographic parity on a subset (no thresholding)."""
    probs = torch.sigmoid(logits)
    s = sens[idx].long()
    if (s == 0).any() and (s == 1).any():
        p0 = probs[idx][s == 0].mean()
        p1 = probs[idx][s == 1].mean()
        return (p0 - p1).pow(2)
    return probs.new_zeros(())

def _soft_eo_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sens: torch.Tensor,
    idx: torch.Tensor,
    mode: str = "tpr",
):
    """
    Soft equal opportunity on a subset (condition on Y=1, no thresholding).
    Returns (E[p̂ | S=0, Y=1] - E[p̂ | S=1, Y=1])^2 where p̂ = sigmoid(logit).
    Gracefully returns 0 if any group lacks positives.
    """
    probs = torch.sigmoid(logits)              # [N]
    idx = idx.long().to(labels.device)
    y_sub = labels[idx].long()                 # [|idx|]
    pos_mask = (y_sub == 1)                    # mask within idx
    if not pos_mask.any():
        return probs.new_zeros(())

    if mode == "tpr":
        idx_pos = idx[pos_mask]                    # indices with Y=1
        s_pos = sens[idx_pos].long()               # sens for positives
        if (s_pos == 0).any() and (s_pos == 1).any():
            p_pos = probs[idx_pos]                 # predicted prob among Y=1
            p0 = p_pos[s_pos == 0].mean()
            p1 = p_pos[s_pos == 1].mean()
            return (p0 - p1).pow(2)
    elif mode == "fpr":
        idx_neg = idx[~pos_mask]                   # indices with Y=0
        s_neg = sens[idx_neg].long()               # sens for negatives
        if (s_neg == 0).any() and (s_neg == 1).any():
            p_neg = probs[idx_neg]                 # predicted prob among Y=0
            p0 = p_neg[s_neg == 0].mean()
            p1 = p_neg[s_neg == 1].mean()
            return (p0 - p1).pow(2)

    return probs.new_zeros(())

def _get_eo_loss(logits, Y, data, idx_tr, eo_mode):
    if eo_mode == "tpr":
        loss_eo  = _soft_eo_from_logits(logits, Y, data.sens, idx_tr, mode="tpr")
    elif eo_mode == "fpr":
        loss_eo  = _soft_eo_from_logits(logits, Y, data.sens, idx_tr, mode="fpr")
    elif eo_mode == "both":
        loss_eo_tpr = _soft_eo_from_logits(logits, Y, data.sens, idx_tr, mode="tpr")
        loss_eo_fpr = _soft_eo_from_logits(logits, Y, data.sens, idx_tr, mode="fpr")
        loss_eo = loss_eo_tpr + loss_eo_fpr
    else:
        raise ValueError(f"Unknown eo_mode: {eo_mode}")
    return loss_eo

def _blend(A_base: SparseTensor, edge_adder: EdgeAdder | None):
    """Return blended SparseTensor A = A_base (+ soft edges if provided)."""
    return (A_base + edge_adder.sparse_tensor()).coalesce() if edge_adder is not None else A_base

def _reduce_losses(loss_list: list[torch.Tensor], method: str = "max", tau: float = 0.5):
    """Hard max or smooth max (log-sum-exp)."""
    if len(loss_list) == 0:
        raise ValueError("Empty loss list.")
    L = torch.stack(loss_list)
    if method == "logsumexp":
        return tau * torch.logsumexp(L / tau, dim=0)
    return torch.max(L, dim=0)[0]

@torch.no_grad()
def _eval_on_graph(backbone, clf, X, A, Y, idx, data):
    H = backbone(X, A)
    logits = clf(H).squeeze(1)
    pred = (logits > 0).long()
    return get_metrics(Y, logits, pred=pred, idx=idx, data=data, neg=False)


def run_edge_adder_unified(args, data, seed_dir):
    """
    Unified trainer for:
      - Baseline edge-adder (single policy using make_cross_group_candidates)
      - Min–max over five policies (policies.build_five_policies)
    Picked by flags: --model edge_adder [baseline] vs --model edge_minmax or --minmax [min–max]
    """
    t0 = time.time()
    device = args.device
    seed = int(seed_dir.split('seed_')[-1])
    X, Y, EI = data.features, data.labels, data.edge_index
    idx_tr, idx_va, idx_te = data.idx_train, data.idx_val, data.idx_test
    in_dim, out_dim = X.size(1), 1

    # backbone + head
    backbone = ConstructModel(in_dim, args.hid_dim, args.encoder, args.layer_num).to(device)
    clf = torch.nn.Linear(args.hid_dim, out_dim).to(device)

    # Build policies of non-edge pair selection
    use_minmax = args.model == "edge_minmax"
    policies: dict[str, EdgeAdder] = {}

    if use_minmax:
        pol_pairs = build_policies(
            X, data.sens, EI,
            policy_names=getattr(args, "policy_names", []),
            k_per_node=getattr(args, "edge_k", 2),
            seed=seed
        )
        for name, pairs in pol_pairs.items():
            if pairs.numel() > 0:
                policies[name] = EdgeAdder(X.size(0), pairs.to(device)).to(device)
    else:
        # Baseline single policy
        cand_ij = make_cross_group_candidates(
            X, data.sens, EI, k_per_node=getattr(args, "edge_k", 2), device=device
        )
        if cand_ij.numel() > 0:
            policies["baseline"] = EdgeAdder(X.size(0), cand_ij.to(device)).to(device)
    t1 = time.time()
    print(f"Policy construction time: {t1 - t0:.2f} seconds.")

    # params & optim
    params = list(backbone.parameters()) + list(clf.parameters())
    for ed in policies.values():
        params += list(ed.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    lam_dp  = float(getattr(args, "lambda_dp", 0.1))
    lam_eo  = float(getattr(args, "lambda_eo", 0.0))
    eo_mode = getattr(args, "eo_mode", "tpr")
    lam_l1  = float(getattr(args, "lambda_edge_l1", 1e-4))
    reduce  =       getattr(args, "max_reduce", "max")
    tau     = float(getattr(args, "lse_tau", 0.5))

    t1 = time.time()
    print(f"Total setup time: {t1 - t0:.2f} seconds. Training for {args.epochs} epochs...")

    pbar = tqdm(range(args.epochs), desc=f"{args.dataset}-{args.encoder}-{'edge_minmax' if use_minmax else 'edge_add'}")
    best = {'score': -1e9, 'state': None}
    elog = EpochLogger(seed_dir, model=("edge_minmax" if use_minmax else "edge_adder"))

    for ep in pbar:
        backbone.train()
        clf.train()
        optimizer.zero_grad()

        if len(policies) == 0:
            raise ValueError("No policies constructed; cannot proceed with edge adder training.")
        else:
            # Compute per-policy losses and reduce by max/logsumexp
            loss_list = []          # total objective per policy (for backprop reduce)
            train_perpol = []       # store components for logging (no grad used for logging)
            for name, ed in policies.items():
                A_blend = _blend(EI, ed)
                H = backbone(X, A_blend)
                logits = clf(H).squeeze(1)
                loss_bce = F.binary_cross_entropy_with_logits(logits[idx_tr], Y[idx_tr].float())
                loss_dp  = _soft_dp_from_logits(logits, data.sens, idx_tr)
                loss_eo  = _get_eo_loss(logits, Y, data, idx_tr, eo_mode)
                loss_l1  = ed.weights().abs().sum()
                loss_total = loss_bce + (lam_dp * loss_dp) + (lam_eo * loss_eo) + (lam_l1 * loss_l1)
                loss_list.append(loss_total)
                train_perpol.append({
                    "policy": name,
                    "loss_bce": float(loss_bce.detach().cpu().item()),
                    "loss_dp":  float(loss_dp.detach().cpu().item()),
                    "loss_eo":  float(loss_eo.detach().cpu().item()),
                    "loss_l1":  float(loss_l1.detach().cpu().item()),
                    "loss_total": float(loss_total.detach().cpu().item()),
                })
            loss = _reduce_losses(loss_list, method=reduce, tau=tau)

        with torch.no_grad():
            # pick the *worst* policy by the training objective for logging components
            loss_tensor = torch.tensor([x["loss_total"] for x in train_perpol], device=args.device)
            worst_tr_idx = int(torch.argmax(loss_tensor).item())
            worst_tr = train_perpol[worst_tr_idx]
            auc_tr, f1_tr, acc_tr, dp_tr, eo_tr = _eval_on_graph(backbone, clf, X, EI, Y, idx_tr, data)
            elog.log(ep, "train", {
                # losses (train)
                "policy":   worst_tr["policy"],
                "loss_total": worst_tr["loss_total"],
                "loss_bce": worst_tr["loss_bce"],
                "loss_dp":  worst_tr["loss_dp"],
                "loss_eo":  worst_tr["loss_eo"],
                "loss_l1":  worst_tr["loss_l1"],
                # metrics (train)
                "auc": auc_tr,
                "f1": f1_tr,
                "acc": acc_tr,
                "dp": dp_tr,
                "eo": eo_tr
            })

        loss.backward()
        optimizer.step()

        # Validation on base graph
        backbone.eval()
        clf.eval()
        with torch.no_grad():
            auc, f1, acc, dp, eo = _eval_on_graph(backbone, clf, X, EI, Y, idx_va, data)

            val_losses = []   # objective per policy (BCE + λ_dp*DP + λ_eo*EO + λ_l1*L1)
            perpol_val = []   # components + metrics per policy (for logging & selection)

            for name, ed in policies.items():
                A_val = _blend(EI, ed)
                H_val = backbone(X, A_val)
                logit_val = clf(H_val).squeeze(1)
                loss_bce_v = F.binary_cross_entropy_with_logits(logit_val[idx_tr], Y[idx_tr].float())
                loss_dp_v  = _soft_dp_from_logits(logit_val, data.sens, idx_tr)
                loss_eo_v  = _get_eo_loss(logit_val, Y, data, idx_tr, eo_mode)
                loss_l1_v  = ed.weights().abs().sum()
                obj_v = loss_bce_v + (lam_dp * loss_dp_v) + (lam_eo * loss_eo_v) + (lam_l1 * loss_l1_v)
                val_losses.append(obj_v)
                perpol_val.append({
                    "policy": name,
                    "loss_bce": float(loss_bce_v.detach().cpu().item()),
                    "loss_dp":  float(loss_dp_v.detach().cpu().item()),
                    "loss_eo":  float(loss_eo_v.detach().cpu().item()),
                    "loss_l1":  float(loss_l1_v.detach().cpu().item()),
                    "loss_total": float(obj_v.detach().cpu().item()),
                })
        worst_obj_idx = int(torch.argmax(torch.stack(val_losses)).item())
        worst_obj = perpol_val[worst_obj_idx]
        worst_obj["auc"] = auc
        worst_obj["f1"]  = f1
        worst_obj["acc"] = acc
        worst_obj["dp"]  = dp
        worst_obj["eo"]  = eo
        elog.log(ep, "val", worst_obj)

        score = (auc + f1) / 2 - dp - eo
        if score > best['score']:
            best['score'] = score
            best['state'] = {'backbone': backbone.state_dict(), 'clf': clf.state_dict()}

        if (ep+1) % args.log_interval == 0:
            pbar.set_postfix(
                loss=f"{float(loss):.3f}",
                loss_bce=f"{worst_tr['loss_bce']:.3f}",
                loss_dp=f"{worst_tr['loss_dp']:.3f}",
                loss_eo=f"{worst_tr['loss_eo']:.3f}",
                loss_l1=f"{worst_tr['loss_l1']:.3f}",
                policy=worst_tr['policy'],
                # auc=f"{auc:.3f}", f1=f"{f1:.3f}", acc=f"{acc:.3f}",
                # dp=f"{dp:.3f}", eo=f"{eo:.3f}"
            )

    # Test on base graph with best checkpoint
    if best['state'] is not None:
        backbone.load_state_dict(best['state']['backbone'])
        clf.load_state_dict(best['state']['clf'])

    backbone.eval()
    clf.eval()
    with torch.no_grad():
        auc_t, f1_t, acc_t, dp_t, eo_t = _eval_on_graph(backbone, clf, X, EI, Y, idx_te, data)

    elog.log(args.epochs, "test", {
        'auc': auc_t, 'f1': f1_t, 'acc': acc_t,
        'dp': dp_t, 'eo': eo_t
    })
    elog.close()

    print(f"[TEST seed={seed}] AUC {auc_t:.4f}  F1 {f1_t:.4f}  ACC {acc_t:.4f}  DP {dp_t:.4f}  EO {eo_t:.4f}")
    return auc_t, f1_t, acc_t, dp_t, eo_t


def run_vanilla(args, data, seed_dir):
    # Basic setup
    seed = int(seed_dir.split('seed_')[-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = data.features
    Y = data.labels
    EI = data.edge_index   # SparseTensor
    idx_tr, idx_va, idx_te = data.idx_train, data.idx_val, data.idx_test
    in_dim = X.shape[1]
    out_dim = 1

    # Build models (backbone + clf)
    backbone = ConstructModel(in_dim, args.hid_dim, args.encoder, args.layer_num).to(device)
    clf = nn.Linear(args.hid_dim, out_dim).to(device)
    params = list(backbone.parameters()) + list(clf.parameters())

    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    pbar = tqdm(range(args.epochs), desc=f"{args.dataset}-{args.encoder}")
    best = {'score': -1e9, 'state': None}
    elog = EpochLogger(seed_dir, model=args.model)

    for ep in pbar:
        backbone.train()
        clf.train()
        opt.zero_grad()

        A_blend = EI

        H = backbone(X, A_blend)          # [N, hid]
        logits = clf(H).squeeze(1)        # [N]
        loss_bce = loss_fn(logits[idx_tr], Y[idx_tr].float())
        loss = loss_bce
        loss_dp, l1 = None, None

        # --- Train metrics & logging ---
        with torch.no_grad():
            pred_tr = (logits > 0).long()
            auc_tr, f1_tr, acc_tr, dp_tr, eo_tr = get_metrics(
                Y, logits, pred=pred_tr, idx=idx_tr, data=data, neg=False
            )
            metrics_train = {
                # losses (train)
                "loss_total": float(loss.item()),
                "loss_bce": float(loss_bce.item()),
                "loss_dp": float(loss_dp.item()) if loss_dp is not None else None,
                "loss_l1": float(l1.item()) if l1 is not None else None,
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
            loss_bce_val = loss_fn(logit_val[idx_va], Y[idx_va].float())
            loss_dp_val = None
            l1_val = None
            loss_val_total = loss_bce_val

        pred_val = (logit_val > 0).long()
        auc, f1, acc, dp, eo = get_metrics(
            Y, logit_val, pred=pred_val, idx=idx_va, data=data, neg=False
        )

        score = (auc + f1) / 2 #- dp - eo
        if score > best['score']:
            best['score'] = score
            best['state'] = {
                'backbone': backbone.state_dict(),
                'clf': clf.state_dict()
            }

        metrics_val = {
            # losses (val)
            'loss_total': float(loss_val_total.item()),
            'loss_bce': float(loss_bce_val.item()),
            'loss_dp': float(loss_dp_val.item()) if loss_dp_val is not None else None,
            'loss_l1': float(l1_val.item()) if l1_val is not None and torch.is_tensor(l1_val) else (float(l1_val) if l1_val is not None else None),
            # metrics (val)
            'auc': auc,
            'f1': f1,
            'acc': acc,
            'dp': dp,
            'eo': eo
        }
        elog.log(ep, "val", metrics_val)

        message = f"loss(bce): {loss.item():.3f}"
        message += f", auc: {auc:.3f}, f1: {f1:.3f}, acc: {acc:.3f}, dp: {dp:.3f}, eo: {eo:.3f}"
        if (ep+1) % args.log_interval == 0:
            pbar.set_postfix({"Metrics": message})

    # --- Test with the best checkpoint ---
    backbone.load_state_dict(best['state']['backbone'])
    clf.load_state_dict(best['state']['clf'])
    backbone.eval()
    clf.eval()
    with torch.no_grad():
        Ht = backbone(X, EI)
        logit_t = clf(Ht).squeeze(1)
    pred_t = (logit_t > 0).long()

    auc_test, f1_test, acc_test, dp_test, eo_test = get_metrics(
        Y, logit_t, pred=pred_t, idx=idx_te, data=data, neg=False
    )
    metrics_test = {
        'auc': auc_test,
        'f1': f1_test,
        'acc': acc_test,
        'dp': dp_test,
        'eo': eo_test
    }
    elog.log(args.epochs, "test", metrics_test)
    elog.close()

    print(f"[TEST] (Seed {seed}) AUC: {auc_test:.4f}  F1: {f1_test:.4f}  ACC: {acc_test:.4f}  DP: {dp_test:.4f}  EO: {eo_test:.4f}")
    return auc_test, f1_test, acc_test, dp_test, eo_test

def load_best_overall_into_args(args):
    """If args.best_overall_path is set, override key HPs from its 'params'."""

    if not getattr(args, "best_overall_path", ""):
        return args

    for best_overall_path in args.best_overall_path:
        with open(best_overall_path, "r") as f:
            obj = json.load(f)
        print("Loaded best overall parameters from:", best_overall_path)

        params = obj.get("params", {})
        for k in params:
            setattr(args, k, params[k])

    # Handle zero regularization flags
    if getattr(args, "use_zero_dp", False):
        setattr(args, "lambda_dp", 0.0)
    if getattr(args, "use_zero_eo", False):
        setattr(args, "lambda_eo", 0.0)

    # Handle special sentinel values
    if getattr(args, "lambda_eo", 0.0) == -1.0:
        print("Setting lambda_eo to lambda_dp")
        setattr(args, "lambda_eo", args.lambda_dp)
    if getattr(args, "layer_num", 0) == 3:
        print("Setting layer_num to 2")
        setattr(args, "layer_num", 2)

    print(args)
    return args

def main(args):
    model_num = 1
    results = Results(args.seed_num, model_num, args)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f'{args.dataset}/{args.encoder}/{args.model}/{ts}'
    args.log_dir = os.path.join(args.log_dir, dir_name)
    use_cuda = torch.cuda.is_available()
    configure_threads(getattr(args, "num_threads", 4))

    data = FairDataset(args.dataset, args.device)
    data.load_data()
    data.info()
    clean_snap = snapshot_clean_data(data)

    for s in range(args.seed_num):
        seed = s + args.start_seed
        set_seed(seed, use_cuda)
        args.seed_dir = os.path.join(args.log_dir, f'seed_{seed}')
        os.makedirs(args.seed_dir, exist_ok=True)

        data = restore_from_snapshot(clean_snap, args.device)

        if getattr(args, "attack", "none") == "nifa":
            def _A_edges(A: SparseTensor) -> int:
                r, c, _ = A.coo()
                return int(r.numel())
            print("[NIFA] Applying node+edge injection attack before training...")
            tic = time.time()
            N0, E0 = int(data.features.size(0)), _A_edges(data.edge_index)
            print(f"[NIFA pre] N={N0}, E={E0}")
            data = apply_nifa_attack(args, data)
            N1, E1 = int(data.features.size(0)), _A_edges(data.edge_index)
            print(f"[NIFA post] N={N1}, E={E1}, ΔN={N1-N0}, ΔE={E1-E0}")
            print(f"✓ attack done in {time.time() - tic:.1f}s")

        if args.model == "vanilla":
            auc, f1, acc, dp, eo = run_vanilla(args, data, args.seed_dir)
        elif args.model == "fairinv":
            pbar = tqdm(total=args.epochs, desc=f"Seed {seed}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
            auc, f1, acc, dp, eo = run_fairinv(args, data, pbar)
        elif args.model in ["edge_adder", "edge_minmax"]:
            auc, f1, acc, dp, eo = run_edge_adder_unified(args, data, args.seed_dir)
        else:
            raise ValueError("Invalid mode.")
        results.auc[s, :], results.f1[s, :], results.acc[s, :], \
            results.parity[s, :], results.equality[s, :] = auc, f1, acc, dp, eo

    results.report_results()
    if args.save_results:
        results.save_results(args)

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    parser = get_parser()
    args = parser.parse_known_args()[0]
    args = load_best_overall_into_args(args)
    main(args)
