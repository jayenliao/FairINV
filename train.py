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

def _soft_dp_from_logits(logits: torch.Tensor, sens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
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

def _blend(A_base: SparseTensor, edge_adder: EdgeAdder | None, num_nodes: int | None = None):
    """Return blended SparseTensor A = A_base (+ soft edges if provided).

    NOTE: Under threat model **B** (eval-time node injection), the evaluation graph may
    have more nodes than the training graph. In that case, EdgeAdder only defines
    edges over the *original* nodes, so we need to pad its sparse tensor to the
    larger (num_nodes, num_nodes) shape before adding it to the attacked graph.
    """
    if edge_adder is None:
        return A_base

    # Fast path: same shape as training graph.
    if num_nodes is None or getattr(edge_adder, "N", None) == num_nodes:
        return (A_base + edge_adder.sparse_tensor()).coalesce()

    # Pad EdgeAdder edges into a larger sparse matrix.
    i, j = edge_adder.cij[0], edge_adder.cij[1]
    w = edge_adder.weights()
    row = torch.cat([i, j], dim=0)
    col = torch.cat([j, i], dim=0)
    val = torch.cat([w, w], dim=0)
    A_ed = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes)).coalesce()
    return (A_base + A_ed).coalesce()


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
    attack_when = getattr(args, 'attack_when', 'train')
    eval_only_attack = (getattr(args, 'attack', 'none') == 'nifa' and attack_when == 'eval')
    clean_snap = snapshot_clean_data(data) if eval_only_attack else None

    in_dim, out_dim = X.size(1), 1

    # backbone + head
    backbone = ConstructModel(in_dim, args.hid_dim, args.encoder, args.layer_num).to(device)
    clf = torch.nn.Linear(args.hid_dim, out_dim).to(device)
    # Optional 2-stage pipeline:
    #   (1) pretrain GNN on base graph without edge-weight L1 (and without learnable edges)
    #   (2) freeze GNN parameters
    #   (3) build cross-group candidates and train EdgeAdder only with edge-weight L1
    edge_pipeline = getattr(args, "edge_pipeline", "joint")
    if edge_pipeline == "freeze_gnn_then_edge":
        lam_dp_full = float(getattr(args, "lambda_dp", 0.0))
        lam_eo_full = float(getattr(args, "lambda_eo", 0.0))
        eo_mode = getattr(args, "eo_mode", "tpr")
        lam_l1 = float(getattr(args, "lambda_edge_l1", 1e-4))
        reduce_method = getattr(args, "max_reduce", "max")
        lse_tau = float(getattr(args, "lse_tau", 0.5))

        pre_epochs = int(getattr(args, "pretrain_epochs", 0) or args.epochs)
        edge_epochs = int(getattr(args, "edge_epochs", 0) or args.epochs)

        # stage-1 fairness coefficients (can override for pretraining)
        lam_dp_pre = getattr(args, "pretrain_lambda_dp", None)
        lam_eo_pre = getattr(args, "pretrain_lambda_eo", None)
        lam_dp_pre = lam_dp_full if lam_dp_pre is None else float(lam_dp_pre)
        lam_eo_pre = lam_eo_full if lam_eo_pre is None else float(lam_eo_pre)

        # Setup common tensors
        X = data.features
        Y = data.labels
        EI = data.edge_index   # SparseTensor
        idx_tr, idx_va, idx_te = data.idx_train, data.idx_val, data.idx_test

        # ---------------- Stage 1: pretrain GNN on base graph ---------------- #
        opt_pre = torch.optim.Adam(
            list(backbone.parameters()) + list(clf.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        best_pre = {'score': -1e9, 'state': None}
        elog = EpochLogger(seed_dir, model=("edge_minmax" if (args.model == "edge_minmax") else "edge_adder"))

        pbar1 = tqdm(range(pre_epochs), desc=f"[stage1-pretrain] seed={seed}", unit="epoch",
                     bar_format="{l_bar}{bar:30}{r_bar}")
        for ep in pbar1:
            backbone.train()
            clf.train()
            opt_pre.zero_grad()

            H = backbone(X, EI)
            logit = clf(H).squeeze(1)

            loss_bce = F.binary_cross_entropy_with_logits(logit[idx_tr], Y[idx_tr].float())
            loss_dp = _soft_dp_from_logits(logit, data.sens, idx_tr) if lam_dp_pre > 0.0 else None
            loss_eo = _get_eo_loss(logit, Y, data, idx_tr, eo_mode) if lam_eo_pre > 0.0 else None

            loss = loss_bce
            if loss_dp is not None:
                loss = loss + (lam_dp_pre * loss_dp)
            if loss_eo is not None:
                loss = loss + (lam_eo_pre * loss_eo)

            loss.backward()
            opt_pre.step()

            # log (pretrain_train): just losses
            elog.log(ep, "pretrain_train", {
                "loss_total": float(loss.item()),
                "loss_bce": float(loss_bce.item()),
                "loss_dp": float(loss_dp.item()) if loss_dp is not None else None,
                "loss_eo": float(loss_eo.item()) if loss_eo is not None else None,
                "loss_l1": None,
                "policy": None,
            })

            # validation on base graph
            backbone.eval()
            clf.eval()
            with torch.no_grad():
                auc, f1, acc, dp, eo = _eval_on_graph(backbone, clf, X, EI, Y, idx_va, data)
                score = (auc + f1) / 2 - dp - eo

            elog.log(ep, "pretrain_val", {
                "auc": auc, "f1": f1, "acc": acc, "dp": dp, "eo": eo,
                "score": score,
                "policy": None,
            })

            if score > best_pre['score']:
                best_pre['score'] = score
                best_pre['state'] = {
                    'backbone': backbone.state_dict(),
                    'clf': clf.state_dict(),
                }

            if (ep + 1) % max(1, int(getattr(args, "log_interval", 20))) == 0:
                pbar1.set_postfix(loss=f"{float(loss):.3f}", score=f"{score:.3f}")

        # Restore best pretrained GNN
        if best_pre['state'] is not None:
            backbone.load_state_dict(best_pre['state']['backbone'])
            clf.load_state_dict(best_pre['state']['clf'])

        # ---------------- Stage 2: freeze GNN params ---------------- #
        for p in backbone.parameters():
            p.requires_grad_(False)
        for p in clf.parameters():
            p.requires_grad_(False)
        backbone.eval()
        clf.eval()

        # ---------------- Stage 3: build candidates & train EdgeAdder ---------------- #
        cand_source = getattr(args, "edge_cand_source", None)
        if cand_source is None:
            cand_source = "emb"  # default for this pipeline
        if cand_source == "emb":
            with torch.no_grad():
                feat_for_cand = backbone(X, EI).detach()
        else:
            feat_for_cand = X.detach()

        use_minmax = (args.model == "edge_minmax")

        if use_minmax:
            _pn = getattr(args, "policy_names", "same_largest,cross_smallest,same_smallest,cross_random,same_random")
            policy_names = [p.strip() for p in str(_pn).split(',') if p.strip()]
            policies_ij = build_policies(
                feat_for_cand, data.sens, EI,
                policy_names=policy_names,
                k_per_node=int(getattr(args, "edge_k", 2)),
                seed=seed
            )
            policies = {name: EdgeAdder(X.size(0), ij, device=device).to(device) for name, ij in policies_ij.items()}
        else:
            cand_ij = make_cross_group_candidates(
                feat_for_cand, data.sens, EI,
                k_per_node=int(getattr(args, "edge_k", 2)),
                device=device
            )
            policies = {"baseline": EdgeAdder(X.size(0), cand_ij, device=device).to(device)}

        # optimizer over EdgeAdder params only
        edge_params = []
        for ed in policies.values():
            edge_params += list(ed.parameters())
        optimizer = torch.optim.Adam(edge_params, lr=args.lr, weight_decay=0.0)

        best = {'score': -1e9, 'state': None}

        alt_rounds = int(getattr(args, "alt_rounds", 0) or 0)
        if alt_rounds > 0 and (not use_minmax):
            alt_edge_epochs = int(getattr(args, "alt_edge_epochs", 0) or edge_epochs or 20)
            alt_gnn_epochs  = int(getattr(args, "alt_gnn_epochs", 0) or pre_epochs or 20)
            alt_gnn_lr = float(getattr(args, "alt_gnn_lr", None) or args.lr)

            def _set_trainable(mod, flag: bool):
                for p in mod.parameters():
                    p.requires_grad_(flag)

            @torch.no_grad()
            def _val_eval(ep: int, phase: str, round_idx: int):
                backbone.eval()
                clf.eval()
                name, ed = next(iter(policies.items()))
                A_val = _blend(EI, ed)
                auc, f1, acc, dp, eo = _eval_on_graph(backbone, clf, X, A_val, Y, idx_va, data)
                score = (auc + f1) / 2 - dp - eo
                row = {
                    "policy": name,
                    "phase": phase,
                    "round": int(round_idx),
                    "auc": auc, "f1": f1, "acc": acc, "dp": dp, "eo": eo,
                    "score": score,
                }
                elog.log(ep, "val", row)
                return float(score), row

            total_steps = alt_rounds * (alt_edge_epochs + alt_gnn_epochs)
            pbar2 = tqdm(range(total_steps), desc=f"[stage3-alt] seed={seed}", unit="epoch")

            global_ep = 0
            for r in range(alt_rounds):
                # -------- (E-step) update edge weights; freeze GNN/clf --------
                _set_trainable(backbone, False)
                _set_trainable(clf, False)
                backbone.eval()
                clf.eval()
                for ed in policies.values():
                    _set_trainable(ed, True)
                    ed.train()

                for _ in range(alt_edge_epochs):
                    ep = pre_epochs + global_ep
                    optimizer.zero_grad(set_to_none=True)

                    name, ed = next(iter(policies.items()))
                    A_tr = _blend(EI, ed)
                    H_tr = backbone(X, A_tr)
                    logit_tr = clf(H_tr).squeeze(1)

                    loss_bce = F.binary_cross_entropy_with_logits(logit_tr[idx_tr], Y[idx_tr].float())
                    loss_dp = _soft_dp_from_logits(logit_tr, data.sens, idx_tr) if lam_dp_full > 0.0 else None
                    loss_eo = _get_eo_loss(logit_tr, Y, data, idx_tr, eo_mode) if lam_eo_full > 0.0 else None
                    loss_l1 = ed.weights().abs().sum()

                    loss = loss_bce
                    if loss_dp is not None:
                        loss = loss + (lam_dp_full * loss_dp)
                    if loss_eo is not None:
                        loss = loss + (lam_eo_full * loss_eo)
                    loss = loss + (lam_l1 * loss_l1)

                    elog.log(ep, "train", {
                        "policy": name,
                        "phase": "edge",
                        "round": int(r),
                        "loss_bce": float(loss_bce.item()),
                        "loss_dp": float(loss_dp.item()) if loss_dp is not None else None,
                        "loss_eo": float(loss_eo.item()) if loss_eo is not None else None,
                        "loss_l1": float(loss_l1.item()),
                        "loss_total": float(loss.item()),
                    })

                    loss.backward()
                    optimizer.step()

                    robust_score, _ = _val_eval(ep, phase="edge", round_idx=r)
                    if robust_score > best["score"]:
                        best["score"] = robust_score
                        best["state"] = {
                            "backbone": backbone.state_dict(),
                            "clf": clf.state_dict(),
                            "policies": {n: ed_.state_dict() for n, ed_ in policies.items()},
                        }

                    global_ep += 1
                    pbar2.update(1)

                # -------- (M-step) update GNN/clf; freeze edge weights --------
                for ed in policies.values():
                    _set_trainable(ed, False)
                    ed.eval()

                _set_trainable(backbone, True)
                _set_trainable(clf, True)
                backbone.train()
                clf.train()

                optimizer_gnn = torch.optim.Adam(
                    list(backbone.parameters()) + list(clf.parameters()),
                    lr=alt_gnn_lr,
                    weight_decay=float(getattr(args, "weight_decay", 0.0)),
                )

                for _ in range(alt_gnn_epochs):
                    ep = pre_epochs + global_ep
                    optimizer_gnn.zero_grad(set_to_none=True)

                    name, ed = next(iter(policies.items()))
                    A_tr = _blend(EI, ed)
                    H_tr = backbone(X, A_tr)
                    logit_tr = clf(H_tr).squeeze(1)

                    loss_bce = F.binary_cross_entropy_with_logits(logit_tr[idx_tr], Y[idx_tr].float())
                    loss_dp = _soft_dp_from_logits(logit_tr, data.sens, idx_tr) if lam_dp_full > 0.0 else None
                    loss_eo = _get_eo_loss(logit_tr, Y, data, idx_tr, eo_mode) if lam_eo_full > 0.0 else None

                    loss = loss_bce
                    if loss_dp is not None:
                        loss = loss + (lam_dp_full * loss_dp)
                    if loss_eo is not None:
                        loss = loss + (lam_eo_full * loss_eo)

                    elog.log(ep, "train", {
                        "policy": name,
                        "phase": "gnn",
                        "round": int(r),
                        "loss_bce": float(loss_bce.item()),
                        "loss_dp": float(loss_dp.item()) if loss_dp is not None else None,
                        "loss_eo": float(loss_eo.item()) if loss_eo is not None else None,
                        "loss_total": float(loss.item()),
                    })

                    loss.backward()
                    optimizer_gnn.step()

                    robust_score, _ = _val_eval(ep, phase="gnn", round_idx=r)
                    if robust_score > best["score"]:
                        best["score"] = robust_score
                        best["state"] = {
                            "backbone": backbone.state_dict(),
                            "clf": clf.state_dict(),
                            "policies": {n: ed_.state_dict() for n, ed_ in policies.items()},
                        }

                    global_ep += 1
                    pbar2.update(1)

            if hasattr(pbar2, "close"):
                pbar2.close()

            # --- Final: restore best and evaluate (supports eval-only attack for threat model B) ---
            if best.get('state', None) is not None:
                backbone.load_state_dict(best['state']['backbone'])
                clf.load_state_dict(best['state']['clf'])
                for _name, _ed in policies.items():
                    if _name in best['state'].get('policies', {}):
                        _ed.load_state_dict(best['state']['policies'][_name])

            backbone.eval()
            clf.eval()
            name0 = list(policies.keys())[0]
            ed0 = policies[name0]
            A_te = _blend(EI, ed0, num_nodes=int(X.size(0)))
            auc_c, f1_c, acc_c, dp_c, eo_c = _eval_on_graph(backbone, clf, X, A_te, Y, idx_te, data)
            row_clean = {
                'policy': name0,
                'phase': 'final',
                'round': int(alt_rounds),
                'auc': auc_c,
                'f1': f1_c,
                'acc': acc_c,
                'dp': dp_c,
                'eo': eo_c,
                'score': (auc_c + f1_c) / 2 - dp_c - eo_c,
            }

            if eval_only_attack:
                # Primary 'test' split is attacked-graph evaluation; keep clean result in 'test_clean'.
                elog.log(global_ep, 'test_clean', row_clean)
                data_att = restore_from_snapshot(clean_snap, device)
                data_att = apply_nifa_attack(args, data_att)
                Xa, Ya, EIa = data_att.features, data_att.labels, data_att.edge_index
                idx_te_a = data_att.idx_test
                A_te_a = _blend(EIa, ed0, num_nodes=int(Xa.size(0)))
                auc_a, f1_a, acc_a, dp_a, eo_a = _eval_on_graph(backbone, clf, Xa, A_te_a, Ya, idx_te_a, data_att)
                row_attack = {
                    'policy': name0,
                    'phase': 'final',
                    'round': int(alt_rounds),
                    'attack': 'nifa',
                    'auc': auc_a,
                    'f1': f1_a,
                    'acc': acc_a,
                    'dp': dp_a,
                    'eo': eo_a,
                    'score': (auc_a + f1_a) / 2 - dp_a - eo_a,
                }
                elog.log(global_ep, 'test', row_attack)
                elog.close()
                print(f"[TEST (clean) seed={seed}] AUC={auc_c:.4f} F1={f1_c:.4f} ACC={acc_c:.4f} DP={dp_c:.4f} EO={eo_c:.4f}")
                print(f"[TEST (eval-attack) seed={seed}] AUC={auc_a:.4f} F1={f1_a:.4f} ACC={acc_a:.4f} DP={dp_a:.4f} EO={eo_a:.4f}")
                return auc_a, f1_a, acc_a, dp_a, eo_a
            else:
                elog.log(global_ep, 'test', row_clean)
                elog.close()
                print(f"[TEST seed={seed}] AUC={auc_c:.4f} F1={f1_c:.4f} ACC={acc_c:.4f} DP={dp_c:.4f} EO={eo_c:.4f}")
                return auc_c, f1_c, acc_c, dp_c, eo_c

        else:
            pbar2 = tqdm(range(edge_epochs), desc=f"[stage3-edge] seed={seed}", unit="epoch",
                         bar_format="{l_bar}{bar:30}{r_bar}")

            epoch_offset = pre_epochs

            for ep2 in pbar2:
                ep = epoch_offset + ep2  # global epoch index for logging
                optimizer.zero_grad()

                # per-policy objectives for training (BCE + λ_dp*DP + λ_eo*EO + λ_l1*L1)
                tr_losses = []
                perpol_tr = []
                for name, ed in policies.items():
                    A_tr = _blend(EI, ed)
                    H_tr = backbone(X, A_tr)
                    logit_tr = clf(H_tr).squeeze(1)

                    loss_bce = F.binary_cross_entropy_with_logits(logit_tr[idx_tr], Y[idx_tr].float())
                    loss_dp = _soft_dp_from_logits(logit_tr, data.sens, idx_tr) if lam_dp_full > 0.0 else None
                    loss_eo = _get_eo_loss(logit_tr, Y, data, idx_tr, eo_mode) if lam_eo_full > 0.0 else None
                    loss_l1 = ed.weights().abs().sum()

                    obj = loss_bce
                    if loss_dp is not None:
                        obj = obj + (lam_dp_full * loss_dp)
                    if loss_eo is not None:
                        obj = obj + (lam_eo_full * loss_eo)
                    obj = obj + (lam_l1 * loss_l1)

                    tr_losses.append(obj)
                    perpol_tr.append({
                        "policy": name,
                        "loss_bce": float(loss_bce.item()),
                        "loss_dp": float(loss_dp.item()) if loss_dp is not None else 0.0,
                        "loss_eo": float(loss_eo.item()) if loss_eo is not None else 0.0,
                        "loss_l1": float(loss_l1.item()),
                    })

                if use_minmax:
                    loss = _reduce_losses(tr_losses, method=reduce_method, tau=lse_tau)
                    worst_tr_idx = int(torch.stack(tr_losses).argmax().item())
                else:
                    loss = tr_losses[0]
                    worst_tr_idx = 0

                worst_tr = perpol_tr[worst_tr_idx]
                worst_tr["loss_total"] = float(loss.item())
                elog.log(ep, "train", worst_tr)

                loss.backward()
                optimizer.step()

                # Validation on BLENDED graph (edge weights are used at inference for this pipeline)
                backbone.eval()
                clf.eval()
                with torch.no_grad():
                    val_rows = []
                    val_scores = []
                    for name, ed in policies.items():
                        A_val = _blend(EI, ed)
                        auc, f1, acc, dp, eo = _eval_on_graph(backbone, clf, X, A_val, Y, idx_va, data)
                        score = (auc + f1) / 2 - dp - eo
                        val_rows.append({
                            "policy": name,
                            "auc": auc, "f1": f1, "acc": acc, "dp": dp, "eo": eo,
                            "score": score,
                        })
                        val_scores.append(score)

                    if use_minmax:
                        worst_val_idx = int(torch.tensor(val_scores).argmin().item())
                        robust_score = float(min(val_scores))
                    else:
                        worst_val_idx = 0
                        robust_score = float(val_scores[0])

                    worst_val = val_rows[worst_val_idx]
                    elog.log(ep, "val", worst_val)

                if robust_score > best['score']:
                    best['score'] = robust_score
                    best['state'] = {
                        'backbone': backbone.state_dict(),
                        'clf': clf.state_dict(),
                        'policies': {n: ed.state_dict() for n, ed in policies.items()}
                    }

                if (ep2 + 1) % max(1, int(getattr(args, "log_interval", 20))) == 0:
                    pbar2.set_postfix(loss=f"{float(loss):.3f}", score=f"{robust_score:.3f}", worst_policy=worst_val.get("policy"))


        # Restore best (including EdgeAdder params)
        if best['state'] is not None:
            backbone.load_state_dict(best['state']['backbone'])
            clf.load_state_dict(best['state']['clf'])
            for n, ed in policies.items():
                if n in best['state']['policies']:
                    ed.load_state_dict(best['state']['policies'][n])

        # Test on BLENDED graph (worst-case over policies if minmax)
        backbone.eval()
        clf.eval()
        with torch.no_grad():
            test_rows = []
            test_scores = []
            for name, ed in policies.items():
                A_te = _blend(EI, ed)
                auc_t, f1_t, acc_t, dp_t, eo_t = _eval_on_graph(backbone, clf, X, A_te, Y, idx_te, data)
                score_t = (auc_t + f1_t) / 2 - dp_t - eo_t
                test_rows.append({
                    "policy": name,
                    "auc": auc_t, "f1": f1_t, "acc": acc_t, "dp": dp_t, "eo": eo_t,
                    "score": score_t,
                })
                test_scores.append(score_t)

            if use_minmax:
                worst_test_idx = int(torch.tensor(test_scores).argmin().item())
            else:
                worst_test_idx = 0
            worst_test = test_rows[worst_test_idx]

        if eval_only_attack:
            # Primary 'test' split is attacked-graph evaluation; keep clean result in 'test_clean'.
            elog.log(epoch_offset + edge_epochs, 'test_clean', worst_test)

            data_att = restore_from_snapshot(clean_snap, device)
            data_att = apply_nifa_attack(args, data_att)
            Xa, Ya, EIa = data_att.features, data_att.labels, data_att.edge_index
            idx_te_a = data_att.idx_test

            test_rows_a, test_scores_a = [], []
            for name, ed in policies.items():
                A_te_a = _blend(EIa, ed, num_nodes=int(Xa.size(0)))
                auc, f1, acc, dp, eo = _eval_on_graph(backbone, clf, Xa, A_te_a, Ya, idx_te_a, data_att)
                score = (auc + f1) / 2 - dp - eo
                row = {'policy': name, 'auc': auc, 'f1': f1, 'acc': acc, 'dp': dp, 'eo': eo, 'score': score}
                test_rows_a.append(row)
                test_scores_a.append(score)

            if use_minmax:
                worst_idx = int(torch.tensor(test_scores_a).argmin().item())
            else:
                worst_idx = 0
            worst_test_a = test_rows_a[worst_idx]
            elog.log(epoch_offset + edge_epochs, 'test', worst_test_a)
            elog.close()

            print(f"[TEST (clean) seed={seed}] policy={worst_test['policy']} AUC={worst_test['auc']:.4f} F1={worst_test['f1']:.4f} ACC={worst_test['acc']:.4f} DP={worst_test['dp']:.4f} EO={worst_test['eo']:.4f}")
            print(f"[TEST (eval-attack) seed={seed}] policy={worst_test_a['policy']} AUC={worst_test_a['auc']:.4f} F1={worst_test_a['f1']:.4f} ACC={worst_test_a['acc']:.4f} DP={worst_test_a['dp']:.4f} EO={worst_test_a['eo']:.4f}")
            return worst_test_a['auc'], worst_test_a['f1'], worst_test_a['acc'], worst_test_a['dp'], worst_test_a['eo']
        else:
            elog.log(epoch_offset + edge_epochs, 'test', worst_test)
            elog.close()

            print(f"[TEST seed={seed}] policy={worst_test['policy']} AUC={worst_test['auc']:.4f} F1={worst_test['f1']:.4f} ACC={worst_test['acc']:.4f} DP={worst_test['dp']:.4f} EO={worst_test['eo']:.4f}")
            return worst_test['auc'], worst_test['f1'], worst_test['acc'], worst_test['dp'], worst_test['eo']

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

    lam_dp  = float(getattr(args, "lambda_dp", 0.0))
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
    lam_dp  = float(getattr(args, "lambda_dp", 0.0))
    lam_eo  = float(getattr(args, "lambda_eo", 0.0))
    eo_mode = getattr(args, "eo_mode", "tpr")

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
        loss_dp  = _soft_dp_from_logits(logits, data.sens, idx_tr)
        loss_eo  = _get_eo_loss(logits, Y, data, idx_tr, eo_mode)
        l1 = None
        loss = loss_bce + (lam_dp * loss_dp) + (lam_eo * loss_eo)

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
                "loss_eo": float(loss_eo.item()) if loss_eo is not None else None,
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
            loss_dp_val = _soft_dp_from_logits(logit_val, data.sens, idx_va) if lam_dp > 0.0 else None
            loss_eo_val = _get_eo_loss(logit_val, Y, data, idx_va, eo_mode) if lam_eo > 0.0 else None
            l1_val = None
            loss_val_total = loss_bce_val

        pred_val = (logit_val > 0).long()
        auc, f1, acc, dp, eo = get_metrics(
            Y, logit_val, pred=pred_val, idx=idx_va, data=data, neg=False
        )

        score = (auc + f1) / 2 - dp - eo
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
            'loss_eo': float(loss_eo_val.item()) if loss_eo_val is not None else None,
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
    metrics_test_clean = {
        'auc': auc_test,
        'f1': f1_test,
        'acc': acc_test,
        'dp': dp_test,
        'eo': eo_test,
    }

    attack_when = getattr(args, 'attack_when', 'train')
    eval_only_attack = (getattr(args, 'attack', 'none') == 'nifa' and attack_when == 'eval')
    if eval_only_attack:
        # Log clean test separately; primary 'test' split will be the attacked-graph result.
        elog.log(args.epochs, 'test_clean', metrics_test_clean)

        snap = snapshot_clean_data(data)
        data_att = restore_from_snapshot(snap, device)
        data_att = apply_nifa_attack(args, data_att)
        Xa, Ya, EIa = data_att.features, data_att.labels, data_att.edge_index
        idx_te_a = data_att.idx_test
        with torch.no_grad():
            Ha = backbone(Xa, EIa)
            logit_a = clf(Ha).squeeze(1)
        pred_a = (logit_a > 0).long()
        auc_a, f1_a, acc_a, dp_a, eo_a = get_metrics(
            Ya, logit_a, pred=pred_a, idx=idx_te_a, data=data_att, neg=False
        )
        metrics_test_attack = {
            'auc': auc_a,
            'f1': f1_a,
            'acc': acc_a,
            'dp': dp_a,
            'eo': eo_a,
        }
        elog.log(args.epochs, 'test', metrics_test_attack)
        elog.close()

        print(f"[TEST (clean)] (Seed {seed}) AUC: {auc_test:.4f}  F1: {f1_test:.4f}  ACC: {acc_test:.4f}  DP: {dp_test:.4f}  EO: {eo_test:.4f}")
        print(f"[TEST (eval-attack)] (Seed {seed}) AUC: {auc_a:.4f}  F1: {f1_a:.4f}  ACC: {acc_a:.4f}  DP: {dp_a:.4f}  EO: {eo_a:.4f}")
        return auc_a, f1_a, acc_a, dp_a, eo_a
    else:
        elog.log(args.epochs, 'test', metrics_test_clean)
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

        attack_when = getattr(args, "attack_when", "train")
        if getattr(args, "attack", "none") == "nifa" and attack_when in ("train", "both"):
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
