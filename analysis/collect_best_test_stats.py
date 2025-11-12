#!/usr/bin/env python3
"""
collect_best_test_stats.py

Given a path to Optuna's study summary `best_overall.json`, load the
best trial directory (stored in best_overall.json["user_attrs"]["trial_dir"]),
locate per-seed metrics.jsonl files, and report the TEST-set mean±std for
{acc, auc, f1, dp, eo} at the epoch that maximizes the *validation* objective.

This script mirrors the objective/value logic used in `tune_optuna.py`
so that "balanced" and attack-oriented objectives choose the same epoch.
"""
from __future__ import annotations

import argparse, json, statistics as stats, os
from pathlib import Path
from typing import Dict, Any, List, Optional

# -----------------------------
# Helpers
# -----------------------------

def fetch_metric(row: Dict[str, Any], base: str):
    """Return row[base] if present, else prefer *val* then *test* suffixed keys.
    This mirrors tune_optuna.fetch_metric.
    """
    if base in row:
        return row[base]
    for suf in ["_val", "_test"]:
        k = base + suf
        if k in row:
            return row[k]
    return None

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def objective_value(
    row: Dict[str, Any], objective: str, balanced_on: str,
    w_dp: float, w_eo: float,
    util_on: str = "f1", util_min: Optional[float] = None, lambda_util: float = 1.0
) -> float:
    """Exact copy of tune_optuna.objective_value (kept in sync)."""
    def _need(keys: list[str]):
        for k in keys:
            if k not in row or row[k] is None:
                raise KeyError(f"objective_value: missing metric '{k}' for objective='{objective}'")

    if objective == "f1":
        v = fetch_metric(row, "f1")
        return float(v) if v is not None else float("-inf")
    if objective == "auc":
        v = fetch_metric(row, "auc")
        return float(v) if v is not None else float("-inf")
    if objective == "auc_f1":
        a = fetch_metric(row, "auc")
        f = fetch_metric(row, "f1")
        if a is None or f is None: return float("-inf")
        return 0.5 * (float(a) + float(f))
    if objective in {"f1_mean_minus_std", "auc_f1_mean_minus_std"}:
        # These are only meaningful across seeds; use plain f1 here to pick epoch.
        v = fetch_metric(row, "f1")
        return float(v) if v is not None else float("-inf")

    # Balanced family
    m = fetch_metric(row, "auc") if balanced_on == "auc" else fetch_metric(row, "f1")
    dp, eo = fetch_metric(row, "dp"), fetch_metric(row, "eo")

    if objective == "balanced":
        if m is None or dp is None or eo is None: return float("-inf")
        return float(m) - w_dp * float(dp) - w_eo * float(eo)

    # Attack-oriented objectives (minimize fairness, optionally constrain utility)
    if objective == "attack_dp_eo":
        _need(["dp", "eo"])
        return w_dp * float(dp) + w_eo * float(eo)
    if objective == "attack_balanced":
        _need(["dp", "eo"])
        u = fetch_metric(row, util_on)
        if u is None:
            raise KeyError(f"objective_value: missing '{util_on}' for attack_balanced")
        score = w_dp * float(dp) + w_eo * float(eo)
        if util_min is not None:
            score -= lambda_util * max(0.0, float(util_min) - float(u))
        return score

    # Default to balanced-like if m/dp/eo exist
    if m is None or dp is None or eo is None:
        return float("-inf")
    return float(m) - w_dp * float(dp) - w_eo * float(eo)

# -----------------------------
# Core
# -----------------------------

def collect_stats(
    trial_dir: Path,
    objective: str, balanced_on: str, w_dp: float, w_eo: float,
    util_on: str = "f1", util_min: Optional[float] = None, lambda_util: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """For each seed under trial_dir/seed_*/, pick the *test* row whose epoch
    matches the best *val* row under the given objective. Aggregate mean/std."""
    metrics = {k: [] for k in ["acc","auc","f1","dp","eo"]}

    seed_dirs = [p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]
    seed_dirs = sorted(seed_dirs, key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name)
    for seed_dir in seed_dirs:
        rows = read_jsonl(seed_dir / "metrics.jsonl")
        if not rows:
            continue
        val_rows = [r for r in rows if r.get("split") == "val"]
        test_rows = [r for r in rows if r.get("split") == "test"]
        if not test_rows:
            continue

        # Choose epoch by validation objective
        best_epoch, best_score = None, float("-inf")
        for r in val_rows:
            try:
                s = objective_value(r, objective, balanced_on, w_dp, w_eo, util_on, util_min, lambda_util)
            except KeyError:
                # If attack objective can't be evaluated on val (missing metrics),
                # gracefully degrade to balanced selection on available metrics.
                s = objective_value(r, "balanced", balanced_on, w_dp, w_eo, util_on, util_min, lambda_util)
            if s > best_score:
                best_score, best_epoch = s, r.get("epoch")

        # Match test row
        chosen = None
        if best_epoch is not None:
            for r in test_rows:
                if r.get("epoch") == best_epoch:
                    chosen = r; break
        if chosen is None:
            chosen = test_rows[-1]

        for k in metrics:
            v = fetch_metric(chosen, k)
            if v is not None:
                metrics[k].append(float(v))

    # Aggregate
    out: Dict[str, Dict[str, float]] = {}
    for k, arr in metrics.items():
        if len(arr) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
        elif len(arr) == 1:
            out[k] = {"mean": arr[0], "std": 0.0, "n": 1, "values": arr}
        else:
            out[k] = {"mean": stats.mean(arr), "std": stats.pstdev(arr), "n": len(arr), "values": arr}
    return out

def print_mean_std(stats_json: Dict[str, Any], show_metric_names: bool = False) -> None:
    for k, v in stats_json.items():
        if k == "acc":
            # keep concise (acc is usually not used in fairness tables)
            continue
        if show_metric_names:
            print(f"{k:>3s}: {v['mean']*100:.2f} ± {v['std']*100:.2f}")
        else:
            print(f"{v['mean']*100:.2f} ± {v['std']*100:.2f}")

def resolve_first_component(p: str) -> Path:
    # Map absolute/relative paths to repo-root-safe first component
    parts = Path(p).parts
    if not parts:
        return Path(".")
    head = parts[0]
    if head in {"", " ", "."} and len(parts) > 1:
        return Path(parts[1])
    return Path(head)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best_json", help="Path to best_overall.json produced by tune_optuna.py")
    ap.add_argument("--model", type=str, default=None, choices=["vanilla", "fairinv", "edge_adder"],
                    help="Optional check that best_overall.json sits under this model.")
    ap.add_argument("--objective", type=str, default="auc_f1",
                    choices=["f1", "auc", "auc_f1", "balanced",
                             "f1_mean_minus_std", "auc_f1_mean_minus_std",
                             "attack_dp_eo", "attack_balanced"])
    ap.add_argument("--balanced_on", default="auc", choices=["auc","f1"])
    ap.add_argument("--w_dp", type=float, default=1.0)
    ap.add_argument("--w_eo", type=float, default=1.0)
    ap.add_argument("--util_on", type=str, default="f1", help="Utility metric for attack_balanced")
    ap.add_argument("--util_min", type=float, default=None, help="Minimum utility target for attack_balanced")
    ap.add_argument("--lambda_util", type=float, default=1.0, help="Penalty weight for falling below util_min")
    ap.add_argument("--out", default=None, help="Where to save the stats JSON (default: next to best_overall.json)")
    ap.add_argument("--show_metric_names", action="store_true", help="Whether to show metric names when printing")
    args = ap.parse_args()

    best = json.load(open(args.best_json))
    trial_dir = best["user_attrs"]["trial_dir"]

    # Make trial_dir robust if the first component differs (e.g., moving logs folder)
    root_json = resolve_first_component(args.best_json)
    root_trial = resolve_first_component(trial_dir)
    if root_json != root_trial:
        trial_dir = str(Path(root_json) / Path(trial_dir).relative_to(root_trial))

    # Try the recorded location; if the timestamp directory changed, fall back to the latest timestamp sibling
    try:
        stats_json = collect_stats(Path(trial_dir), args.objective, args.balanced_on, args.w_dp, args.w_eo,
                                   args.util_on, args.util_min, args.lambda_util)
    except FileNotFoundError:
        # Replace the timestamp component by the latest available one under .../<objective>/
        parts = Path(trial_dir).parts
        if len(parts) >= 1:
            ts_old = parts[-2]
            parent = Path(trial_dir).parent.parent
            cand = sorted([p.name for p in parent.iterdir() if p.is_dir()])
            if cand:
                ts_new = cand[-1]
                repaired = str(Path(trial_dir).as_posix().replace(f"/{ts_old}/", f"/{ts_new}/"))
                stats_json = collect_stats(Path(repaired), args.objective, args.balanced_on, args.w_dp, args.w_eo,
                                           args.util_on, args.util_min, args.lambda_util)
            else:
                raise

    out_path = Path(args.best_json).with_name("best_trial_test_stats.json") if args.out is None else Path(args.out)
    with open(out_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print_mean_std(stats_json, show_metric_names=args.show_metric_names)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
