from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Sequence

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

    nifa_node: Optional[List[int]] = None
    nifa_edge: Optional[List[int]] = None
    nifa_ratio: Optional[List[float]] = None
    nifa_gamma: Optional[List[float]] = None

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
        )


class AdvGraphPool:
    """
    Generates attacked graph variants using NIFA.
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
        out = []
        for i in range(int(self.cfg.k)):
            seed_i = self.seed + i * int(self.cfg.seed_stride) + int(epoch)
            d = self.restore_fn(self.clean_snapshot, self.device)
            if self.cfg.attack == "nifa":
                a = self._variant_args(i)
                with temp_seed(seed_i):
                    d = apply_nifa_attack(a, d)
            out.append(d)
        return out
