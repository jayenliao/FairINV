from types import SimpleNamespace
import torch
import dgl
from torch_sparse import SparseTensor

from nifa_attack import Attacker  # uses nifa_model.GCN internally
# NOTE: nifa_attack imports nifa_model (patched below)

def _sparse_edge_count(A: SparseTensor) -> int:
    r, c, _ = A.coo()
    return int(r.numel())

def _to_dgl_from_fair(data):
    """Convert FairDataset (PyG/SparseTensor) -> DGLGraph with required node data."""
    N = data.features.size(0)
    # Use the explicit edge_index_nor (2, E) for DGL construction
    ei = data.edge_index_nor.cpu()
    g = dgl.graph((ei[0], ei[1]), num_nodes=N)
    g.ndata['feature']   = data.features.detach().cpu()
    g.ndata['label']     = data.labels.detach().cpu().long()
    g.ndata['sensitive'] = data.sens.detach().cpu().long()
    return g

def _update_fair_from_dgl(data, g, device):
    """Write back attacked graph to the FairDataset-like structure in-place."""
    # features / labels / sens
    data.features = g.ndata['feature'].to(device)
    data.labels   = g.ndata['label'].to(device).long()
    data.sens     = g.ndata['sensitive'].to(device).float()

    # edges -> both SparseTensor and raw edge_index (PyG-style)
    src, dst = g.edges()
    edge_index_nor = torch.stack([src, dst], dim=0).to(device).long()
    data.edge_index_nor = edge_index_nor  # (2, E)

    data.edge_index = SparseTensor.from_edge_index(
        edge_index_nor, sparse_sizes=(g.num_nodes(), g.num_nodes())
    ).coalesce()
    return data

def apply_nifa_attack(args, data):
    """
    1) Convert FairINV data -> DGL
    2) Run NIFA attacker
    3) Convert attacked graph -> back to FairINV data object
    """
    # --- before stats (on the FairINV side)
    N0 = int(data.features.size(0))
    E0 = _sparse_edge_count(data.edge_index)

    device = args.device
    g = _to_dgl_from_fair(data)
    g = g.to(device)

    in_dim  = int(data.features.size(1))
    out_dim = 2  # binary cls; CE loss in NIFA code expects logits for 2 classes

    # Build the index split dict expected by NIFA
    split = {
        'train_index': data.idx_train.cpu().long(),
        'val_index':   data.idx_val.cpu().long(),
        'test_index':  data.idx_test.cpu().long(),
    }

    # Map main args -> the minimal NIFA args Attacker needs
    nifa = SimpleNamespace(
        T=args.nifa_T, theta=args.nifa_theta,
        node=args.nifa_node, edge=args.nifa_edge,
        alpha=args.nifa_alpha, beta=args.nifa_beta,
        ratio=args.nifa_ratio, mode=args.nifa_mode,
        epochs=args.nifa_epochs, lr=args.nifa_lr, loops=args.nifa_loops,
        hid_dim=args.hid_dim
    )

    # NIFA trains a Bayesian net inside; needs autograd.
    with torch.enable_grad():
        attacker = Attacker(g, in_dim, args.hid_dim, out_dim, device, nifa)
        g_poison, _unc = attacker.attack(g, split)

    # IMPORTANT: keep original train/val/test indices (on original nodes).
    # The injected nodes have label = -1 and are *not* in the split;
    # they only affect message passing (as intended).
    data = _update_fair_from_dgl(data, g_poison, device)
    N1 = int(data.features.size(0))
    E1 = _sparse_edge_count(data.edge_index)

    injected_nodes = int((data.labels < 0).logical_and(data.sens < 0).sum().item())
    print(f"[NIFA] Nodes {N0} -> {N1}  (+{N1-N0}) | "
          f"Edges {E0} -> {E1}  (+{E1-E0}) | "
          f"Injected node markers found: {injected_nodes}")

        # hard sanity checks
    assert N1 >= N0, "Node count decreased after attack — impossible."
    assert E1 >= E0, "Edge count did not increase after attack — check Edge_Attack."
    assert injected_nodes == (N1 - N0), \
        "Injected nodes not marked with label=-1 & sensitive=-1; verify Edge/Feature_Attack."

    # splits must stay on original nodes
    assert int(data.idx_train.max()) < N0
    assert int(data.idx_val.max())   < N0
    assert int(data.idx_test.max())  < N0

    return data
