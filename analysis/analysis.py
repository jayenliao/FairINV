import numpy as np
import pandas as pd
import argparse, json, re, os
from pathlib import Path

METRIC_KEYS = ["dp_mean", "eo_mean", "f1_mean", "auc_mean"]
MAP_METRIC_TO_LABEL = {
    "DP": "Demographic Parity",
    "EO": "Equal Opportunity",
    "F1": "F1-Score",
    "AUC": "AUROC",
}

def collect_results_among_seeds(
    exp_dir: str,
    pattern: str = "results_among_*_seeds.json",
    verbose: bool = False,
):
    """Collect aggregated JSON results (e.g., results_among_10_seeds.json) under an experiment directory.

    Expected layout (most common):
        exp_dir/<method>/<dataset>/<encoder>/<model>/<timestamp>/results_among_*_seeds.json

    This function is robust to exp_dir depth by reading metadata from the last 5 folders above the JSON file.
    It also tolerates JSONs that either:
        - have a top-level key "results" (preferred), or
        - store metrics directly at the top level.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir.resolve()}")
    rows = []

    for p in exp_dir.rglob(pattern):
        try:
            rel = p.relative_to(exp_dir)
        except Exception:
            rel = None

        # metadata from directory names (closest parents)
        timestamp = p.parent.name
        model = p.parent.parent.name
        encoder = p.parent.parent.parent.name
        dataset = p.parent.parent.parent.parent.name
        method = rel.parts[0] if (rel is not None and len(rel.parts) > 0) else p.parent.parent.parent.parent.parent.name

        obj = _load_json(str(p), verbose=verbose)
        if obj is None:
            continue
        res = obj.get("results", obj)
        if not isinstance(res, dict):
            if verbose:
                print(f"[skip] unexpected json format: {p}")
            continue

        row = dict(res)
        row["method"] = method
        row["dataset"] = dataset
        row["encoder"] = encoder
        row["model"] = model
        row["timestamp"] = timestamp

        # parse seed count from filename if present
        m = re.search(r"results_among_(\d+)_seeds", p.name)
        if m:
            row["n_seeds"] = int(m.group(1))

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        if verbose:
            print(f"[empty] No files matched {pattern} under {exp_dir}")
        return df

    # keep a nice order if columns exist
    front = [c for c in ["method", "dataset", "encoder", "model", "timestamp", "n_seeds"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values(by=[c for c in front if c in df.columns])
    return df

def _read_metrics_file(path: Path) -> pd.DataFrame:
    """Read metrics.jsonl or metrics.csv into a DataFrame."""
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
        return pd.DataFrame(rows)
    elif path.suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def _extract_run_meta(path, exp_dir):
    """
    Parse run metadata from a log path.

    Expected layout (example):
      <exp_dir>/<method>/<dataset>/<encoder>/<model>/<timestamp>/seed_<k>/metrics.(jsonl|csv)

    Returns a dict with keys:
      method, dataset, encoder, model, timestamp, seed
    Missing fields are set to None.
    """
    from pathlib import Path
    import re

    p = Path(path).resolve()
    root = Path(exp_dir).resolve()

    try:
        rel = p.relative_to(root)
    except Exception:
        rel = p

    parts = list(rel.parts)

    seed = None
    seed_idx = None
    for i, s in enumerate(parts):
        if s.startswith("seed_"):
            seed_idx = i
            try:
                seed = int(s.split("_", 1)[1])
            except Exception:
                seed = s.split("_", 1)[1] if "_" in s else s
            break

    base = parts[:seed_idx] if seed_idx is not None else parts[:-1]  # drop filename if no seed
    if not base:
        return dict(method=None, dataset=None, encoder=None, model=None, timestamp=None, seed=seed)

    # find timestamp like YYYYMMDD-HHMMSS, otherwise use last folder as timestamp
    ts_pat = re.compile(r"^\d{8}-\d{6}$")
    ts_idx = None
    for i in range(len(base) - 1, -1, -1):
        if ts_pat.match(base[i]):
            ts_idx = i
            break
    if ts_idx is None:
        ts_idx = len(base) - 1

    timestamp = base[ts_idx] if ts_idx >= 0 else None
    model = base[ts_idx - 1] if ts_idx - 1 >= 0 else None
    encoder = base[ts_idx - 2] if ts_idx - 2 >= 0 else None
    dataset = base[ts_idx - 3] if ts_idx - 3 >= 0 else None
    method = base[ts_idx - 4] if ts_idx - 4 >= 0 else None

    return dict(method=method, dataset=dataset, encoder=encoder, model=model, timestamp=timestamp, seed=seed)


def _read_split_record_from_metrics_jsonl(jsonl_path, split_name="test_clean"):
    """
    Read metrics.jsonl and return the 'best' record for the given split.
    Best = highest epoch; tie-breaker = latest timestamp; fallback = last occurrence.
    Returns None if no matching split is found.
    """
    import json
    from datetime import datetime

    def _ts(v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return None
        s = str(v)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return None

    best = None
    best_key = None  # (epoch, ts, idx)
    idx = -1

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("split") != split_name:
                continue

            epoch = rec.get("epoch")
            try:
                epoch = int(epoch) if epoch is not None else -1
            except Exception:
                epoch = -1

            ts = _ts(rec.get("timestamp"))
            key = (epoch, ts or datetime.min, idx)

            if best is None or key > best_key:
                best = rec
                best_key = key

    return best


def _read_split_record_from_metrics_csv(csv_path, split_name="test_clean"):
    """
    Read metrics.csv and return the 'best' row for the given split as a dict.
    Best = highest epoch; tie-breaker = latest timestamp; fallback = last occurrence.
    Returns None if no matching split is found.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        return None

    df = df[df["split"] == split_name].copy()
    if df.empty:
        return None

    # epoch sorting if available
    if "epoch" in df.columns:
        df["__epoch"] = pd.to_numeric(df["epoch"], errors="coerce").fillna(-1).astype(int)
    else:
        df["__epoch"] = -1

    # timestamp sorting if available
    if "timestamp" in df.columns:
        df["__ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["__ts"] = df["__ts"].fillna(pd.Timestamp.min)
    else:
        df["__ts"] = pd.Timestamp.min

    df["__idx"] = range(len(df))
    df = df.sort_values(["__epoch", "__ts", "__idx"], ascending=True)
    row = df.iloc[-1].to_dict()

    for k in ["__epoch", "__ts", "__idx"]:
        row.pop(k, None)
    return row


def collect_test_clean_metrics_from_seeds(exp_dir, split_name="test_clean", prefer=("jsonl", "csv")):
    """
    Crawl <exp_dir>/**/seed_*/metrics.(jsonl|csv), extract `split_name` (default: test_clean),
    and return a DataFrame with metadata columns + metric columns.

    prefer: order to prefer file types when both exist for the same seed directory.
    """
    from pathlib import Path
    import pandas as pd

    root = Path(exp_dir)
    jsonl_files = list(root.rglob("seed_*/metrics.jsonl"))
    csv_files = list(root.rglob("seed_*/metrics.csv"))

    by_seed_dir = {}
    for p in jsonl_files:
        by_seed_dir.setdefault(p.parent, {})["jsonl"] = p
    for p in csv_files:
        by_seed_dir.setdefault(p.parent, {})["csv"] = p

    rows = []
    for seed_dir, files in sorted(by_seed_dir.items(), key=lambda x: str(x[0])):
        chosen = None
        for typ in prefer:
            if typ in files:
                chosen = (typ, files[typ])
                break
        if chosen is None:
            continue

        typ, fpath = chosen
        if typ == "jsonl":
            rec = _read_split_record_from_metrics_jsonl(fpath, split_name=split_name)
        else:
            rec = _read_split_record_from_metrics_csv(fpath, split_name=split_name)

        if rec is None:
            continue

        meta = _extract_run_meta(fpath, exp_dir=root)
        # merge meta + record; keep split for traceability, but you can drop it later if desired
        row = {}
        row.update(meta)
        row.update(rec)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # best-effort numeric conversion for metric columns (keep meta as-is)
    meta_cols = {"method", "dataset", "encoder", "model", "timestamp", "seed", "split"}
    for c in df.columns:
        if c in meta_cols:
            continue
        df[c] = pd.to_numeric(df[c])

    return df


def aggregate_over_seeds(df, group_cols=None, seed_col="seed"):
    """
    Aggregate per-seed metrics into mean/std/count over seeds.

    Returns a DataFrame whose metric columns are expanded to:
      <metric>_mean, <metric>_std, <metric>_count
    """
    import pandas as pd
    import numpy as np

    if df is None or df.empty:
        return df

    if group_cols is None:
        group_cols = [c for c in ["method", "dataset", "encoder", "model"] if c in df.columns]

    metric_cols = []
    for c in df.columns:
        if c in set(group_cols) | {seed_col, "split"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            metric_cols.append(c)

    if not metric_cols:
        # no numeric metrics found; still return distinct groups with seed count
        out = (
            df.groupby(group_cols, dropna=False)[seed_col]
            .nunique()
            .reset_index()
            .rename(columns={seed_col: "n_seeds"})
        )
        return out

    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and aggregate results across seeds.")
    parser.add_argument("exp_dir", type=str, help="Experiment directory to crawl for results.")
    parser.add_argument("--split_name", type=str, default="test_clean", help="Split name to extract from metrics files.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the aggregated results as CSV.")
    args = parser.parse_args()

    # Step 1: Collect per-seed metrics for the specified split
    df_seed = collect_test_clean_metrics_from_seeds(args.exp_dir, split_name=args.split_name)

    # Step 2: Aggregate over seeds
    df_agg = aggregate_over_seeds(df_seed)

    if args.output is not None and args.output != "auto":
        exp_dir = Path(args.output).parent
        exp_dir.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(args.output, index=False)
        print(f"Aggregated results saved to {args.output}")
    elif args.output == "auto":
        print(f"Experiment directory: {args.exp_dir}")
        exp_dir = Path(args.exp_dir.replace("logs", "analysis"))
        print(f"Analysis directory: {exp_dir}")
        os.makedirs(exp_dir, exist_ok=True)
        auto_path = exp_dir / f"aggregated_results_{args.split_name}.csv"
        df_agg.to_csv(auto_path, index=False)
        print(f"Aggregated results saved to {auto_path}")
    else:
        exp_dir = None
        print(df_agg)

    # Step 3: Get pivot tables
    contains_10_seeds = df_agg["epoch_count"] == 10
    if not contains_10_seeds.all():
        print("\033[93mWarning: Not all groups have 10 seeds. Check 'epoch_count' column for details.\033[0m")

    # pts_attacked = {}
    # for metric in METRIC_KEYS:
    #     pt = df_agg[contains_10_seeds].pivot_table(
    #         index=["dataset", "method"],
    #         columns=["encoder"],
    #         values=metric,
    #         observed=False
    #     )
    #     pts_attacked[metric] = pt
    #     print(f"\nPivot table for {metric} (mean):")
    #     print(pt)
    # if exp_dir is not None:
    #     fn = exp_dir / "pivot_tables_attacked.json"
    #     with open(fn, "w", encoding="utf-8") as f:
    #         json.dump({k: v.to_dict() for k, v in pts_attacked.items()}, f, indent=2)
    #     print(f"Pivot tables saved to {fn}")


