#!/usr/bin/env python3
import os, argparse, json, statistics as stats
from pathlib import Path

def fetch_metric(row, base):
    if base in row:
        return row[base]
    for suf in ["_test", "_val"]:
        k = base + suf
        if k in row:
            return row[k]
    return None

def read_jsonl(p: Path):
    if not p.exists(): return []
    rows = []
    with p.open() as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    return rows

def objective_value(row, objective, balanced_on, w_dp, w_eo):
    if objective == "f1":
        v = fetch_metric(row, "f1")
        return float(v) if v is not None else float("-inf")
    if objective == "auc":
        v = fetch_metric(row, "auc")
        return float(v) if v is not None else float("-inf")
    m = fetch_metric(row, "auc") if balanced_on == "auc" else fetch_metric(row, "f1")
    dp, eo = fetch_metric(row, "dp"), fetch_metric(row, "eo")
    if m is None or dp is None or eo is None: return float("-inf")
    return float(m) - w_dp * float(dp) - w_eo * float(eo)

def collect_stats(trial_dir: Path, objective: str, balanced_on: str, w_dp: float, w_eo: float):
    metrics = {k: [] for k in ["acc","auc","f1","dp","eo"]}
    for seed_dir in sorted([p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        rows = read_jsonl(seed_dir / "metrics.jsonl")
        val_rows = [r for r in rows if r.get("split") == "val"]
        test_rows = [r for r in rows if r.get("split") == "test"]
        if not test_rows: continue
        # Try to align test with best val epoch; else fallback to last test
        best_epoch = None
        best_score = float("-inf")
        for r in val_rows:
            s = objective_value(r, objective, balanced_on, w_dp, w_eo)
            if s > best_score:
                best_score, best_epoch = s, r.get("epoch")
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
                try: metrics[k].append(float(v))
                except: pass

    summary = {}
    for k, arr in metrics.items():
        if len(arr) == 0: continue
        mu = stats.mean(arr)
        sd = stats.pstdev(arr) if len(arr) > 1 else 0.0
        summary[k] = {"mean": mu, "std": sd, "n": len(arr), "values": arr}
    return summary

def print_mean_std(stats_json, show_metric_names:bool=False) -> None:
    for k, v in stats_json.items():
        k_ = f"{k}:"
        if show_metric_names:
            print(f"{k_:4s} {v['mean']*100:.2f} ± {v['std']*100:.2f}")
        else:
            if k == "acc":
                continue
            print(f"{v['mean']*100:.2f} ± {v['std']*100:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("best_json", help="Path to best_overall.json produced by tune_optuna.py")
    ap.add_argument("--model", type=str, default=None, choices=["vanilla", "fairinv", "edge_adder"], help="")
    ap.add_argument("--objective", type=str, default="auc_f1",
                    choices=["f1", "auc", "auc_f1", "balanced", "f1_mean_minus_std", "auc_f1_mean_minus_std"])
    ap.add_argument("--balanced_on", default="auc", choices=["auc","f1"])
    ap.add_argument("--w_dp", type=float, default=1.0)
    ap.add_argument("--w_eo", type=float, default=1.0)
    ap.add_argument("--out", default=None, help="Where to save the stats JSON (default: next to best_overall.json)")
    ap.add_argument("--show_metric_names", action="store_true", help="Whether to show metric names in printed output")
    args = ap.parse_args()

    def _get_first_layer_name(s:str) -> Path:
        out = s.split('/')[0]
        out = s.split('/')[1] if out in ['', ' ', '.'] else out
        return Path(out)

    best = json.load(open(args.best_json))
    root_json = _get_first_layer_name(args.best_json)
    trial_dir = best["user_attrs"]["trial_dir"]
    root_trial = _get_first_layer_name(trial_dir)
    if root_json != root_trial:
        trial_dir = str(Path(root_json) / Path(trial_dir).relative_to(root_trial))
    try:
        stats_json = collect_stats(Path(trial_dir), args.objective, args.balanced_on, args.w_dp, args.w_eo)
    except FileNotFoundError:
        ts_old = trial_dir.split('/')[-2]
        lst_ts = os.listdir(Path(trial_dir).parent.parent)
        lst_ts.sort()
        ts_new = lst_ts[-1]
        trial_dir = trial_dir.replace(ts_old, ts_new)
        stats_json = collect_stats(Path(trial_dir), args.objective, args.balanced_on, args.w_dp, args.w_eo)

    if args.out is None:
        out_path = Path(args.best_json).with_name("best_trial_test_stats.json")
    else:
        out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print_mean_std(stats_json, show_metric_names=args.show_metric_names)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
