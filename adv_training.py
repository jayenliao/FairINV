from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence

import torch
from torch_sparse import SparseTensor

from nifa_bridge import apply_nifa_attack
from utils import temp_seed


def reduce_tensor_list(loss_list: Sequence[torch.Tensor], method: str = "max", tau: float = 0.5) -> torch.Tensor:
    if len(loss_list) == 0:
        raise ValueError("reduce_tensor_list got empty list")
    L = torch.stack(list(loss_list))
    if method == "mean":
        return L.mean()
    if method == "logsumexp":
        tau = float(tau)
        return tau * torch.logsumexp(L / tau, dim=0)
    return torch.max(L, dim=0)[0]


def _broadcast_list(values: Optional[Sequence], k: int, name: str):
    if values is None:
        return None
    values = list(values)
    if len(values) == k:
        return values
    if len(values) == 1:
        return values * k
    raise ValueError(f"{name} must have length 1 or {k}, got {len(values)}")


def project_box_budget(w: torch.Tensor, w_max: float = 1.0, budget: float = -1.0) -> torch.Tensor:
    """Project weights to 0<=w<=w_max, optionally with sum(w)<=budget.

    Note: budget projection uses a cheap scaling projection after clamping, which is
    stable and usually sufficient for PGD-style inner maximization.
    """
    w = w.clamp_(0.0, float(w_max))
    if budget is None:
        return w
    budget = float(budget)
    if budget < 0:
        return w
    s = w.sum()
    if s <= budget:
        return w
    return w * (budget / (s + 1e-12))


def build_added_edge_sparse(num_nodes: int, cand_pairs_ij: torch.LongTensor, w: torch.Tensor) -> SparseTensor:
    """Undirected SparseTensor with values=w on candidate pairs.

    cand_pairs_ij: LongTensor [2, M] with undirected pairs (i,j) (not duplicated)
    w: Tensor [M] in float (may require_grad)
    """
    if cand_pairs_ij.numel() == 0:
        return SparseTensor(sparse_sizes=(num_nodes, num_nodes)).to(w.device)

    i, j = cand_pairs_ij[0], cand_pairs_ij[1]
    # duplicate for undirected
    row = torch.cat([i, j], dim=0)
    col = torch.cat([j, i], dim=0)
    val = torch.cat([w, w], dim=0)
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes)).coalesce()


def pgd_maximize_weights(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    w0: torch.Tensor,
    steps: int = 5,
    step_size: float = 0.1,
    use_sign: bool = True,
    w_max: float = 1.0,
    budget: float = -1.0,
) -> torch.Tensor:
    """Maximize loss_fn(w) over w with PGD, returning w_adv (detached)."""
    steps = int(steps)
    if steps <= 0:
        return project_box_budget(w0.detach(), w_max=w_max, budget=budget).detach()

    w = w0.detach()
    w = project_box_budget(w, w_max=w_max, budget=budget).detach().requires_grad_(True)

    for _ in range(steps):
        loss = loss_fn(w)
        grad = torch.autograd.grad(loss, w, only_inputs=True, retain_graph=False, create_graph=False)[0]
        if use_sign:
            grad = grad.sign()
        w = w + float(step_size) * grad
        w = project_box_budget(w, w_max=w_max, budget=budget).detach().requires_grad_(True)

    return w.detach()


@dataclass
class AdvTrainConfig:
    enabled: bool
    attack: str = "nifa"
    mode: str = "mix"
    k: int = 1
    gen: str = "precompute"
    refresh: int = 0
    cache_device: bool = False

    mix_lambda: float = 1.0
    include_clean: bool = False
    reduce: str = "max"
    tau: float = 0.5
    seed_stride: int = 1000

    # NIFA (graph-variant adversary)
    nifa_node: Optional[List[int]] = None
    nifa_edge: Optional[List[int]] = None
    nifa_ratio: Optional[List[float]] = None
    nifa_gamma: Optional[List[float]] = None

    # Edge-weight adversary (inner max over weights on a fixed candidate edge set)
    edge_steps: int = 5
    edge_step_size: float = 0.1
    edge_init: str = "rand"      # rand | zero
    edge_use_sign: bool = True   # PGD sign(grad) vs raw grad
    edge_budget: float = -1.0    # <0 disables sum(w) constraint
    edge_w_max: float = 1.0      # box: 0<=w<=edge_w_max
    edge_policy: str = "same_smallest"
    edge_k_per_node: int = 0     # 0 => fall back to args.edge_k

    @classmethod
    def from_args(cls, args) -> "AdvTrainConfig":
        k = int(getattr(args, "advtrain_k", 1))
        return cls(
            enabled=bool(getattr(args, "advtrain", False)),
            attack=str(getattr(args, "advtrain_attack", "nifa")),
            mode=str(getattr(args, "advtrain_mode", "mix")),
            k=k,
            gen=str(getattr(args, "advtrain_gen", "precompute")),
            refresh=int(getattr(args, "advtrain_refresh", 0) or 0),
            cache_device=bool(getattr(args, "advtrain_cache_device", False)),
            mix_lambda=float(getattr(args, "advtrain_mix_lambda", 1.0)),
            include_clean=bool(getattr(args, "advtrain_include_clean", False)),
            reduce=str(getattr(args, "advtrain_reduce", "max")),
            tau=float(getattr(args, "advtrain_tau", 0.5)),
            seed_stride=int(getattr(args, "advtrain_seed_stride", 1000)),
            nifa_node=_broadcast_list(getattr(args, "advtrain_nifa_node", None), k, "advtrain_nifa_node"),
            nifa_edge=_broadcast_list(getattr(args, "advtrain_nifa_edge", None), k, "advtrain_nifa_edge"),
            nifa_ratio=_broadcast_list(getattr(args, "advtrain_nifa_ratio", None), k, "advtrain_nifa_ratio"),
            nifa_gamma=_broadcast_list(getattr(args, "advtrain_nifa_gamma", None), k, "advtrain_nifa_gamma"),
            edge_steps=int(getattr(args, "advtrain_edge_steps", 5)),
            edge_step_size=float(getattr(args, "advtrain_edge_step_size", 0.1)),
            edge_init=str(getattr(args, "advtrain_edge_init", "rand")),
            edge_use_sign=(str(getattr(args, "advtrain_edge_grad", "sign")).lower() != "raw"),
            edge_budget=float(getattr(args, "advtrain_edge_budget", -1.0)),
            edge_w_max=float(getattr(args, "advtrain_edge_w_max", 1.0)),
            edge_policy=str(getattr(args, "advtrain_edge_policy", "same_smallest")),
            edge_k_per_node=int(getattr(args, "advtrain_edge_k", 0) or 0),
        )


class AdvGraphPool:
    """Generates attacked graph variants using NIFA.

    Expects the caller to provide snapshot_clean_data/restore_from_snapshot on their side.
    """

    def __init__(self, args, cfg: AdvTrainConfig, clean_snapshot: dict, restore_fn, seed: int, device):
        self.args = args
        self.cfg = cfg
        self.clean_snapshot = clean_snapshot
        self.restore_fn = restore_fn
        self.seed = int(seed)
        self.device = device

        self._cache = None
        self._last_refresh = None

        if self.cfg.enabled and self.cfg.gen == "precompute":
            self._cache = self._gen(epoch=0)

    def get_variants(self, epoch: int):
        if not self.cfg.enabled:
            return []
        if self.cfg.gen == "precompute":
            return list(self._cache)

        if self.cfg.refresh <= 0:
            if self._cache is None:
                self._cache = self._gen(epoch=epoch)
            return list(self._cache)

        if self._last_refresh is None or (epoch - self._last_refresh) >= self.cfg.refresh:
            self._cache = self._gen(epoch=epoch)
            self._last_refresh = int(epoch)
        return list(self._cache)

    def _variant_args(self, i: int):
        ns = SimpleNamespace(**vars(self.args))
        if self.cfg.nifa_node is not None:
            ns.nifa_node = int(self.cfg.nifa_node[i])
        if self.cfg.nifa_edge is not None:
            ns.nifa_edge = int(self.cfg.nifa_edge[i])
        if self.cfg.nifa_ratio is not None:
            ns.nifa_ratio = float(self.cfg.nifa_ratio[i])
        if self.cfg.nifa_gamma is not None:
            ns.nifa_gamma = float(self.cfg.nifa_gamma[i])
        return ns

    def _gen(self, epoch: int):
        if self.cfg.attack != "nifa":
            raise ValueError(f"AdvGraphPool only supports attack='nifa', got {self.cfg.attack}")
        out = []
        for i in range(int(self.cfg.k)):
            seed_i = self.seed + i * int(self.cfg.seed_stride) + int(epoch)
            d = self.restore_fn(self.clean_snapshot, self.device)
            a = self._variant_args(i)
            with temp_seed(seed_i):
                d = apply_nifa_attack(a, d)
            out.append(d)
        return out
