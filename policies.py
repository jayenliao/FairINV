import math, random
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

@torch.no_grad()
def _coo_edge_set(A: SparseTensor):
    row, col, _ = A.coo()
    return set(zip(row.cpu().tolist(), col.cpu().tolist()))

@torch.no_grad()
def _topk_pairs_blocked(X_src: Tensor, X_tgt: Tensor, idx_src: Tensor, idx_tgt: Tensor,
                        k: int, mode: str):
    """
    Return two 1-D LongTensors (i_list, j_list) of size ~= len(idx_src)*k
    selecting top-k per src under cosine similarity.
    mode ∈ {"largest","smallest"}
    """
    device = X_src.device
    # l2-normalize
    Xs = F.normalize(X_src[idx_src], p=2, dim=1).cpu()
    Xt = F.normalize(X_tgt[idx_tgt], p=2, dim=1).cpu()

    B = 4096  # block rows to bound memory
    i_all, j_all = [], []
    for b in range(0, Xs.size(0), B):
        Xb = Xs[b:b+B]                      # [b, d]
        S = Xb @ Xt.T                       # [b, |tgt|] cosine
        if mode == "largest":
            topv, topk = torch.topk(S, k=min(k, S.size(1)), dim=1, largest=True)
        elif mode == "smallest":
            # For smallest similarity, flip sign and take largest
            topv, topk = torch.topk(-S, k=min(k, S.size(1)), dim=1, largest=True)
        else:
            raise ValueError("mode must be largest/smallest")
        src_pick = idx_src[b:b+B].repeat_interleave(topk.size(1))
        tgt_pick = idx_tgt[topk.reshape(-1)]
        i_all.append(src_pick)
        j_all.append(tgt_pick)

    i_cat = torch.cat(i_all).to(device)
    j_cat = torch.cat(j_all).to(device)
    # remove duplicates
    pairs = torch.stack([i_cat, j_cat], dim=0)
    pairs = torch.unique(pairs, dim=1)
    return pairs[0], pairs[1]

@torch.no_grad()
def _filter_non_edges(i: Tensor, j: Tensor, Aset: set):
    keep = []
    for a, b in zip(i.tolist(), j.tolist()):
        if (a, b) not in Aset and (b, a) not in Aset and a != b:
            keep.append((a, b))
    if not keep:
        return (torch.empty(0, dtype=torch.long, device=i.device),
                torch.empty(0, dtype=torch.long, device=i.device))
    ii, jj = zip(*keep)
    return torch.tensor(ii, dtype=torch.long, device=i.device), \
           torch.tensor(jj, dtype=torch.long, device=i.device)

@torch.no_grad()
def _random_pairs(idx_a: Tensor, idx_b: Tensor, per_node: int, Aset: set, device):
    i_all, j_all = [], []
    for u in idx_a.tolist():
        # sample without replacement
        cand = [v for v in idx_b.tolist()
                if (u, v) not in Aset and (v, u) not in Aset and v != u]
        if not cand:
            continue
        m = min(per_node, len(cand))
        pick = random.sample(cand, m)
        i_all.extend([u]*m)
        j_all.extend(pick)
    if not i_all:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))
    return torch.tensor(i_all, device=device), torch.tensor(j_all, device=device)

@torch.no_grad()
def build_policies(
    features: Tensor, sens: Tensor, A: SparseTensor,
    policy_names: list = ["same_largest", "cross_smallest", "same_smallest", "cross_random", "same_random"],
    k_per_node: int = 2, seed: int = 0,
    node_subset: Tensor | None = None
) -> dict:
    """
    Returns dict name -> pairs LongTensor [2, M].
    Policies:
      p1: same-group largest similarity
      p2: cross-group smallest similarity
      p3: same-group smallest similarity
      p4: random cross-group
      p5: random same-group
    """
    if len(policy_names) == 0:
        raise ValueError("policy_names cannot be empty")

    random.seed(seed)
    device = features.device
    s = sens.long().cpu()
    idx0 = torch.where(s == 0)[0]
    idx1 = torch.where(s == 1)[0]
    Aset = _coo_edge_set(A)
    if node_subset is not None:
        node_subset = node_subset.detach().cpu().long()
        mask = torch.zeros(features.size(0), dtype=torch.bool)
        mask[node_subset] = True
        idx0 = idx0[mask[idx0]]
        idx1 = idx1[mask[idx1]]
        
    out = {}

    # p1: same-group largest similarity
    policy_name = "same_largest"
    if policy_name in policy_names and len(idx0) and len(idx1):
        i1a, j1a = _topk_pairs_blocked(features, features, idx0, idx0, k_per_node, "largest")
        i1b, j1b = _topk_pairs_blocked(features, features, idx1, idx1, k_per_node, "largest")
        i1 = torch.cat([i1a, i1b]); j1 = torch.cat([j1a, j1b])
        i1, j1 = _filter_non_edges(i1, j1, Aset)
        out[policy_name] = torch.stack([i1, j1], dim=0)
    else:
        out[policy_name] = torch.empty(2, 0, dtype=torch.long, device=device)

    # p2: cross-group smallest similarity
    policy_name = "cross_smallest"
    if policy_name in policy_names and len(idx0) and len(idx1):
        i2a, j2a = _topk_pairs_blocked(features, features, idx0, idx1, k_per_node, "smallest")
        i2b, j2b = _topk_pairs_blocked(features, features, idx1, idx0, k_per_node, "smallest")
        i2 = torch.cat([i2a, i2b]); j2 = torch.cat([j2a, j2b])
        i2, j2 = _filter_non_edges(i2, j2, Aset)
        out[policy_name] = torch.stack([i2, j2], dim=0)
    else:
        out[policy_name] = torch.empty(2, 0, dtype=torch.long, device=device)

    # p3: same-group smallest similarity
    policy_name = "same_smallest"
    if policy_name in policy_names and len(idx0) and len(idx1):
        i3a, j3a = _topk_pairs_blocked(features, features, idx0, idx0, k_per_node, "smallest")
        i3b, j3b = _topk_pairs_blocked(features, features, idx1, idx1, k_per_node, "smallest")
        i3 = torch.cat([i3a, i3b]); j3 = torch.cat([j3a, j3b])
        i3, j3 = _filter_non_edges(i3, j3, Aset)
        out[policy_name] = torch.stack([i3, j3], dim=0)
    else:
        out[policy_name] = torch.empty(2, 0, dtype=torch.long, device=device)

    # p4: random cross-group
    policy_name = "cross_random"
    if policy_name in policy_names and len(idx0) and len(idx1):
        i4a, j4a = _random_pairs(idx0, idx1, k_per_node, Aset, device)
        i4b, j4b = _random_pairs(idx1, idx0, k_per_node, Aset, device)
        i4 = torch.cat([i4a, i4b]); j4 = torch.cat([j4a, j4b])
        out[policy_name] = torch.stack([i4, j4], dim=0)
    else:
        out[policy_name] = torch.empty(2, 0, dtype=torch.long, device=device)

    # p5: random same-group
    policy_name = "same_random"
    if policy_name in policy_names:
        i5a, j5a = _random_pairs(idx0, idx0, k_per_node, Aset, device) if len(idx0) else (torch.empty(0, dtype=torch.long, device=device),)*2
        i5b, j5b = _random_pairs(idx1, idx1, k_per_node, Aset, device) if len(idx1) else (torch.empty(0, dtype=torch.long, device=device),)*2
        i5 = torch.cat([i5a, i5b]); j5 = torch.cat([j5a, j5b])
        out[policy_name] = torch.stack([i5, j5], dim=0) if i5.numel() else torch.empty(2, 0, dtype=torch.long, device=device)
    else:
        out[policy_name] = torch.empty(2, 0, dtype=torch.long, device=device)

    return out
