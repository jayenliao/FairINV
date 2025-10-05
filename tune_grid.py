#!/usr/bin/env python3
"""
tune.py — Simple, robust hyperparameter tuning harness for this codebase.

- Grid search over user-specified lists (e.g., --lr_list 1e-2 5e-3 1e-3).
- Multi-seed evaluation per trial.
- Selection objective: F1, AUC, or a balanced combo (AUC - w_dp*DP - w_eo*EO, or F1 - w_dp*DP - w_eo*EO).
- Picks best trial by mean VAL objective across seeds (default).
- Organized outputs under logs/tune/.
"""

import argparse, json, os, time, math, itertools, copy, hashlib
from pathlib import Path
from typing import Dict, Any

import torch # type: ignore
from args import get_args as get_base_args
from data import FairDataset
from utils import set_seed
from train import run as run_vanilla_or_edge, run_fairinv

def now_ts():
    return time.strftime("%Y%m%d-%H%M%S")

def hash_of(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def fetch_metric(obj: Dict[str, Any], base: str):
    # Robust across FairINV (auc/dp/eo) and vanilla/edge_adder (auc_val/dp_val/eo_val on VAL)
    if base in obj:
        return obj[base]
    for suffix in ["_val", "_test"]:
        k = base + suffix
        if k in obj:
            return obj[k]
    return None

def parse_seed_jsonl(seed_dir: Path):
    jl = seed_dir / "metrics.jsonl"
    val_rows, test_rows = [], []
    if not jl.exists():
        return val_rows, test_rows
    with jl.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("split") == "val":
                val_rows.append(obj)
            elif obj.get("split") == "test":
                test_rows.append(obj)
    return val_rows, test_rows

def objective_from_row(row: Dict[str, Any], objective: str, balanced_on: str, w_dp: float, w_eo: float):
    if objective == "f1":
        return as_float(fetch_metric(row, "f1"), default=float("-inf"))
    elif objective == "auc":
        return as_float(fetch_metric(row, "auc"), default=float("-inf"))
    elif objective == "balanced":
        m = fetch_metric(row, "auc") if balanced_on == "auc" else fetch_metric(row, "f1")
        dp, eo = fetch_metric(row, "dp"), fetch_metric(row, "eo")
        if m is None or dp is None or eo is None:
            return float("-inf")
        return float(m) - w_dp * float(dp) - w_eo * float(eo)
    else:
        raise ValueError("objective must be {'f1','auc','balanced'}")

def summarize_trial(trial_dir: Path, objective: str, balanced_on: str, w_dp: float, w_eo: float):
    per_seed = []
    for seed_dir in sorted([p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        val_rows, test_rows = parse_seed_jsonl(seed_dir)
        # best VAL
        best_val_score, best_val_epoch, best_val_metrics = float("-inf"), None, {}
        for r in val_rows:
            score = objective_from_row(r, objective, balanced_on, w_dp, w_eo)
            if score > best_val_score:
                best_val_score = score
                best_val_epoch = r.get("epoch")
                best_val_metrics = {
                    "auc": fetch_metric(r, "auc"),
                    "f1":  fetch_metric(r, "f1"),
                    "acc": fetch_metric(r, "acc"),
                    "dp":  fetch_metric(r, "dp"),
                    "eo":  fetch_metric(r, "eo"),
                    "loss": fetch_metric(r, "loss_val") or fetch_metric(r, "loss"),
                }
        # TEST = last logged test
        test_row = test_rows[-1] if test_rows else {}
        test_score = objective_from_row(test_row, objective, balanced_on, w_dp, w_eo)
        test_metrics = {
            "auc": fetch_metric(test_row, "auc"),
            "f1":  fetch_metric(test_row, "f1"),
            "acc": fetch_metric(test_row, "acc"),
            "dp":  fetch_metric(test_row, "dp"),
            "eo":  fetch_metric(test_row, "eo"),
        }
        per_seed.append({
            "seed": int(seed_dir.name.split("_")[-1]),
            "best_val_score": best_val_score,
            "best_val_epoch": best_val_epoch,
            "best_val_metrics": best_val_metrics,
            "test_score": test_score,
            "test_metrics": test_metrics,
        })

    if not per_seed:
        return {"per_seed": [], "val_mean": None, "test_mean": None}

    val_mean = sum(s["best_val_score"] for s in per_seed) / len(per_seed)
    test_scores = [s["test_score"] for s in per_seed if s["test_score"] is not None and not math.isnan(s["test_score"])]
    test_mean = sum(test_scores) / len(test_scores) if test_scores else None
    return {"per_seed": per_seed, "val_mean": val_mean, "test_mean": test_mean}

def build_parser():
    base = get_base_args()  # for defaults
    p = argparse.ArgumentParser(description="Hyperparameter tuning harness")

    # Core model/data
    p.add_argument("--model", choices=["vanilla", "fairinv", "edge_adder"], default=base.model)
    p.add_argument("--encoder", choices=["gcn","gat","gin","sage","sgc"], default=base.encoder)
    p.add_argument("--dataset", choices=["nba", "bail", "pokec_z", "pokec_n", "german"], default=base.dataset)

    # Training basics
    p.add_argument("--epochs", type=int, default=base.epochs)
    p.add_argument("--weight_decay", type=float, default=base.weight_decay)
    p.add_argument("--lr", type=float, default=base.lr)
    p.add_argument("--hid_dim", type=int, default=base.hid_dim)
    p.add_argument("--layer_num", type=int, default=base.layer_num)
    p.add_argument("--dropout", type=float, default=base.dropout)
    p.add_argument("--log_root", type=str, default="logs/tune")
    p.add_argument("--log_interval", type=int, default=base.log_interval)

    # Seeds
    p.add_argument("--seeds", type=int, nargs="+", default=[base.start_seed + i for i in range(max(1, base.seed_num or 1))])
    p.add_argument("--start_seed", type=int, default=base.start_seed)
    p.add_argument("--seed_num", type=int, default=base.seed_num or 1)

    # FairINV-only
    p.add_argument("--alpha", type=float, default=base.alpha)
    p.add_argument("--lr_sp", type=float, default=base.lr_sp)
    p.add_argument("--env_num", type=int, default=base.env_num)
    p.add_argument("--partition_times", type=int, default=base.partition_times)

    # EdgeAdder-only
    p.add_argument("--edge_k", type=int, default=base.edge_k)
    p.add_argument("--lambda_dp", type=float, default=base.lambda_dp)
    p.add_argument("--lambda_edge_l1", type=float, default=base.lambda_edge_l1)

    # Search spaces (lists). If omitted, scalars are used.
    for k, typ in [
        ("lr_list", float), ("weight_decay_list", float), ("hid_dim_list", int), ("layer_num_list", int),
        ("dropout_list", float), ("alpha_list", float), ("lr_sp_list", float), ("env_num_list", int),
        ("edge_k_list", int), ("lambda_dp_list", float), ("lambda_edge_l1_list", float),
    ]:
        p.add_argument(f"--{k}", type=typ, nargs="+")

    # Objective
    p.add_argument("--objective", choices=["f1","auc","balanced"], default="balanced",
                   help="Selection objective computed on VAL split.")
    p.add_argument("--balanced_on", choices=["auc","f1"], default="auc",
                   help="When objective=balanced, choose the utility backbone.")
    p.add_argument("--w_dp", type=float, default=1.0, help="Weight on DP.")
    p.add_argument("--w_eo", type=float, default=1.0, help="Weight on EO.")

    p.add_argument("--tag", type=str, default="", help="Optional tag appended to sweep folder name.")
    return p

def iter_grid(args):
    space = {
        "lr": args.lr_list if args.lr_list else [args.lr],
        "weight_decay": args.weight_decay_list if args.weight_decay_list else [args.weight_decay],
        "hid_dim": args.hid_dim_list if args.hid_dim_list else [args.hid_dim],
        "layer_num": args.layer_num_list if args.layer_num_list else [args.layer_num],
        "dropout": args.dropout_list if args.dropout_list else [args.dropout],
    }
    if args.model == "fairinv":
        space.update({
            "alpha": args.alpha_list if args.alpha_list else [args.alpha],
            "lr_sp": args.lr_sp_list if args.lr_sp_list else [args.lr_sp],
            "env_num": args.env_num_list if args.env_num_list else [args.env_num],
        })
    if args.model == "edge_adder":
        space.update({
            "edge_k": args.edge_k_list if args.edge_k_list else [args.edge_k],
            "lambda_dp": args.lambda_dp_list if args.lambda_dp_list else [args.lambda_dp],
            "lambda_edge_l1": args.lambda_edge_l1_list if args.lambda_edge_l1_list else [args.lambda_edge_l1],
        })
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))

def build_trial_dir(root: Path, args, cfg: Dict[str, Any]) -> Path:
    meta = {"dataset": args.dataset, "encoder": args.encoder, "model": args.model, "objective": args.objective, **cfg}
    h = hash_of(meta)
    base = root / args.dataset / args.encoder / args.model / args.objective
    base = base / (now_ts() + (f"_{args.tag}" if args.tag else ""))
    return base / f"trial_{h}"

def main():
    p = build_parser()
    args = p.parse_args()

    if args.seeds is None or len(args.seeds) == 0:
        args.seeds = [args.start_seed + i for i in range(max(1, args.seed_num))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device == torch.device("cuda")
    ds = FairDataset(args.dataset, device)
    ds.load_data()

    sweep_root = Path(args.log_root)
    summary_rows = []

    for cfg in iter_grid(args):
        trial_dir = build_trial_dir(sweep_root, args, cfg)
        ensure_dir(trial_dir)

        with (trial_dir / "args.json").open("w") as f:
            json.dump({"base": {k:(str(v) if isinstance(v, torch.device) else v) for k,v in vars(args).items()}, "cfg": cfg}, f, indent=2)

        # run all seeds
        for seed in args.seeds:
            a = copy.deepcopy(args)
            for k, v in cfg.items():
                setattr(a, k, v)
            a.start_seed = seed
            a.seed_num = 1
            a.device = device
            a.cuda = torch.cuda.is_available()
            a.log_dir = str(trial_dir)
            a.seed_dir = str(trial_dir / f"seed_{seed}")

            set_seed(seed, use_cuda=use_cuda)
            os.makedirs(a.seed_dir, exist_ok=True)
            if a.model == "fairinv":
                run_fairinv(a, ds)
            else:
                run_vanilla_or_edge(a, ds, a.seed_dir)

        summ = summarize_trial(trial_dir, args.objective, args.balanced_on, args.w_dp, args.w_eo)
        with (trial_dir / "trial_summary.json").open("w") as f:
            json.dump(summ, f, indent=2)

        row = {"trial_dir": str(trial_dir), **cfg, "val_mean": summ.get("val_mean"), "test_mean": summ.get("test_mean")}
        summary_rows.append(row)

    if summary_rows:
        import csv
        csv_path = sweep_root / f"tune_summary_{args.dataset}_{args.encoder}_{args.model}_{args.objective}_{now_ts()}.csv"
        ensure_dir(csv_path.parent)
        headers = sorted(set().union(*[row.keys() for row in summary_rows]))
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
            for r in summary_rows: w.writerow(r)

        best = max(summary_rows, key=lambda r: (r["val_mean"] if r["val_mean"] is not None else float("-inf")))
        with (sweep_root / "best_overall.json").open("w") as f:
            json.dump(best, f, indent=2)
        print(f"\033[31m[tune] Wrote sweep CSV: {csv_path}\033[0m")
        best['test_mean'] = best['test_mean'] if best['test_mean'] is not None else float('nan')
        print(f"\033[31m[tune] Best (VAL): {best['trial_dir']}  val_mean={best['val_mean']:.4f}  test_mean={best['test_mean']:.4f}\033[0m")
    else:
        print("[tune] No trials executed. Check your grid.")

if __name__ == "__main__":
    main()
