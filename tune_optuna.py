"""
tune_optuna.py — Optuna-based hyperparameter tuner for Vanilla, FairINV, and EdgeAdder
across GCN/GAT/GIN/SAGE/SGC backbones and german/bail/pokec_z/pokec_n/nba datasets.

- One study per (model, encoder, dataset) scenario.
- Multi-seed evaluation per trial; objective is the mean VAL score across seeds.
- Selection objective: F1, AUC, or balanced (AUC−w_dp·DP−w_eo·EO or F1−...).
- Organized outputs under logs/optuna/...
- Writes: best_overall.json for each study, and an Optuna dataframe CSV.

Notes:
- We reuse existing training entrypoints: train.run() and train.run_fairinv().
- We parse seed-level metrics from metrics.jsonl emitted by logger.EpochLogger.
- Pruning is disabled by default because training runs are monolithic; enable if you
  refactor train loops to report intermediate values from within Optuna trials.
"""

import argparse, json, os, time, math, copy, hashlib
import statistics as stats
from pathlib import Path
from typing import Dict, Any, Tuple, List
from tqdm import tqdm

import torch  # type: ignore
import optuna

from args import get_args as get_base_args
from data import FairDataset
from utils import set_seed
from utils import configure_threads
from train import run_vanilla, run_edge_adder_unified, run_fairinv, load_best_overall_into_args

# -----------------------------
# Small utils
# -----------------------------

def now_ts():
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def md5_of(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def fetch_metric(obj: Dict[str, Any], base: str):
    if base in obj:
        return obj[base]
    for suf in ["_val", "_test"]:
        k = base + suf
        if k in obj:
            return obj[k]
    return None

def objective_value(
    row: Dict[str, Any], objective: str, balanced_on: str,
    w_dp: float, w_eo: float,
    util_on: str = "f1", util_min: float | None = None, lambda_util: float = 1.0
) -> float:
    if objective == "f1":
        v = fetch_metric(row, "f1")
        return float(v) if v is not None else float("-inf")
    if objective == "auc":
        v = fetch_metric(row, "auc")
        return float(v) if v is not None else float("-inf")
    if objective == "auc_f1":
        v_auc = fetch_metric(row, "auc")
        v_f1 = fetch_metric(row, "f1")
        return float(v_auc + v_f1) * 0.5 if v_auc is not None and v_f1 is not None else float("-inf")
    # balanced
    m = fetch_metric(row, "auc") if balanced_on == "auc" else fetch_metric(row, "f1")
    dp, eo = fetch_metric(row, "dp"), fetch_metric(row, "eo")
    if m is None or dp is None or eo is None:
        return float("-inf")

    if objective == "balanced":
        return float(m) - w_dp * float(dp) - w_eo * float(eo)
    if objective == "attack_dp_eo":
        return w_dp * float(dp) + w_eo * float(eo)
    if objective == "attack_balanced":
        u = fetch_metric(row, util_on)
        if u is None: return float("-inf")
        score = w_dp * float(dp) + w_eo * float(eo)
        if util_min is not None:
            score -= lambda_util * max(0.0, float(util_min) - float(u))
            if float(u) < float(util_min):  # optional hard fail:
                # still return penalized score so trials remain comparable
                return score
        return score
    return float(m) - w_dp * float(dp) - w_eo * float(eo)

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

def summarize_trial_dir(trial_dir: Path, objective: str, balanced_on: str, w_dp: float, w_eo: float):
    per_seed = []
    for sd in sorted([p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        rows = read_jsonl(sd / "metrics.jsonl")
        val_rows = [r for r in rows if r.get("split") == "val"]
        test_rows = [r for r in rows if r.get("split") == "test"]
        # best val row by objective
        best = None
        best_val = float("-inf")
        for r in val_rows:
            v = objective_value(r, objective, balanced_on, w_dp, w_eo)
            if v > best_val:
                best_val = v
                best = r
        test_row = test_rows[-1] if len(test_rows) > 0 else {}
        per_seed.append({
            "seed": int(sd.name.split("_")[-1]),
            "best_val_score": best_val,
            "best_val_row": best,
            "test_row": test_row,
            "test_score": objective_value(test_row, objective, balanced_on, w_dp, w_eo),
        })
    if not per_seed:
        return {
            "val_mean": None, "test_mean": None, "per_seed": [],
            "val_metric_stats": {"f1_mean": None, "f1_std": None, "auc_mean": None, "auc_std": None}
        }

    val_mean = sum(s["best_val_score"] for s in per_seed) / len(per_seed)
    test_scores = [s["test_score"] for s in per_seed if s["test_score"] is not None and not math.isnan(s["test_score"])]
    test_mean = sum(test_scores) / len(test_scores) if test_scores else None

    # after building per_seed
    f1_vals, auc_vals = [], []
    for s in per_seed:
        r = s.get("best_val_row") or {}
        v_f1 = fetch_metric(r, "f1")
        v_auc = fetch_metric(r, "auc")
        if v_f1 is not None: f1_vals.append(float(v_f1))
        if v_auc is not None: auc_vals.append(float(v_auc))

    def _mean_std(xs):
        if not xs: return None, None
        mu = stats.mean(xs)
        sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
        return mu, sd

    f1_mean, f1_std = _mean_std(f1_vals)
    auc_mean, auc_std = _mean_std(auc_vals)

    return {
        "val_mean": val_mean,
        "test_mean": test_mean,
        "per_seed": per_seed,
        "val_metric_stats": {
            "f1_mean": f1_mean, "f1_std": f1_std,
            "auc_mean": auc_mean, "auc_std": auc_std,
        }
    }

# -----------------------------
# Scenario-specific search spaces
# -----------------------------

SMALL = {"german", "bail", "nba"}
LARGE = {"pokec_z", "pokec_n"}

def _suggest_nifa_hparams(trial: optuna.trial.Trial, dataset: str) -> Dict[str, Any]:
    """
    NIFA-only hyperparameter search space (dataset-aware).
    Grounded in the original NIFA settings/ablations:
      - b (injected nodes) ~1% of labeled on large graphs; a bit wider on small graphs
      - d (edges per injected node) ≈ dataset's avg degree (Pokec ~50)
      - k% in {0.1, 0.25, 0.5, 0.75} with 0.5 often best
      - α in {0.005..0.2}, β in {2,4,8,16}
      - T ~20 (search a small band), loops ~20 on Pokec; ~10–15 on smaller graphs
    """
    hp: Dict[str, Any] = {}

    # --- Shared knobs (paper-faithful) ---
    # Uncertainty percentile (top-k%) prefers 0.5; search over paper's discrete set
    hp["nifa_theta"]  = trial.suggest_categorical("nifa_theta", [0.1, 0.25, 0.5, 0.75])
    # Loss weights
    hp["nifa_alpha"]  = trial.suggest_categorical("nifa_alpha", [0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    hp["nifa_beta"]   = trial.suggest_categorical("nifa_beta",  [2, 4, 8, 16])
    # MC-dropout samples T (paper uses 20; allow a narrow search)
    hp["nifa_T"]      = trial.suggest_int("nifa_T", 16, 28, step=2)
    # Outer loops (a.k.a. max_iter vibe)
    # Pokec*: ~20; small graphs can use 8–15
    loops_large = trial.suggest_int("nifa_loops_large", 15, 25)  # used for Pokec
    loops_small = trial.suggest_int("nifa_loops_small", 8, 15)
    # LR for surrogate/feature optim in attack (paper ~1e-3; log search around it)
    hp["nifa_lr"]     = trial.suggest_float("nifa_lr", 1e-4, 5e-3, log=True)
    # How many of the high-uncertainty nodes to target when wiring edges
    hp["nifa_ratio"]  = trial.suggest_float("nifa_ratio", 0.25, 0.75)
    # Target-node selector: paper’s main (uncertainty) plus the degree variant (Appendix)
    hp["nifa_mode"]   = trial.suggest_categorical("nifa_mode", ["uncertainty", "degree"])

    # --- Dataset-aware b (=nifa_node) and d (=nifa_edge), plus loops ---
    if dataset == "pokec_z":
        # b around 1% of labeled nodes; Table A1 reports b≈102; search a tight band
        hp["nifa_node"] = trial.suggest_int("nifa_node", 60, 200, step=10)
        # d near avg degree ~50; search a narrow window
        hp["nifa_edge"] = trial.suggest_int("nifa_edge", 40, 65, step=1)
        hp["nifa_loops"] = loops_large
    elif dataset == "pokec_n":
        # Table A1 b≈87; similar band
        hp["nifa_node"] = trial.suggest_int("nifa_node", 50, 180, step=10)
        hp["nifa_edge"] = trial.suggest_int("nifa_edge", 40, 65, step=1)
        hp["nifa_loops"] = loops_large
    elif dataset == "nba":
        # Small graph: keep b small (≈1–5% of labeled → single digits/teens)
        hp["nifa_node"] = trial.suggest_int("nifa_node", 4, 20, step=2)
        # avg degree typically much lower than Pokec
        hp["nifa_edge"] = trial.suggest_int("nifa_edge", 8, 20, step=1)
        hp["nifa_loops"] = loops_small
    elif dataset == "bail":
        hp["nifa_node"] = trial.suggest_int("nifa_node", 10, 100, step=5)
        hp["nifa_edge"] = trial.suggest_int("nifa_edge", 6, 24, step=1)
        hp["nifa_loops"] = loops_small
    elif dataset == "german":
        hp["nifa_node"] = trial.suggest_int("nifa_node", 10, 80, step=5)
        hp["nifa_edge"] = trial.suggest_int("nifa_edge", 6, 20, step=1)
        hp["nifa_loops"] = loops_small
    else:
        # sensible fallbacks
        hp["nifa_node"] = trial.suggest_int("nifa_node", 20, 100, step=5)
        hp["nifa_edge"] = trial.suggest_int("nifa_edge", 8, 24, step=1)
        hp["nifa_loops"] = loops_small

    return hp

def suggest_hparams(trial: optuna.trial.Trial, model: str, encoder: str, dataset: str,
                    tune_scope: str = "gnn", attack: str = "none") -> Dict[str, Any]:
    """Return a dict of suggested hyperparams for this (model, encoder, dataset)."""
    hp: Dict[str, Any] = {}

    # If only tuning attack, return NIFA hparams directly
    if tune_scope in {"attack","both"} and attack == "nifa":
        hp.update(_suggest_nifa_hparams(trial, dataset))
        if tune_scope == "attack":
            return hp

    # Common spaces: learning rate, weight decay, dropout, hidden size, layers
    if dataset in SMALL:
        hp["lr"] = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        hp["hid_dim"] = trial.suggest_categorical("hid_dim", [16, 32, 64])
    else:  # LARGE
        hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
        hp["hid_dim"] = trial.suggest_categorical("hid_dim", [32, 64, 128])

    hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.7)
    # layer_num: used by GCN/GAT; SGC ignores beyond 1; GIN/SAGE custom modules ignore layer_num internally.
    if model == "fairinv":
        if encoder in ["gcn", "gin", "sgc"]:
            hp["layer_num"] = 1
        elif encoder in ["sage"]:
            hp["layer_num"] = 1
        else: # gat
            hp["layer_num"] = 1
    else:
        if encoder in {"gcn", "gat"}:
            hp["layer_num"] = 1
        elif encoder == "sgc":
            hp["layer_num"] = 1  # ConstructModel stacks a single SGConv
        else: # sage, gin
            hp["layer_num"] = 1

    # Model-specific
    if model == "fairinv":
        hp["alpha"] = trial.suggest_categorical("alpha", [0.001, 0.01, 0.1, 0.5, 1, 10, 100])
        hp["lr_sp"] = trial.suggest_categorical("lr_sp", [0.01, 0.05, 0.1, 0.5])
        # hp["alpha"] = trial.suggest_float("alpha", 1e-1, 1e+1, log=True)       # balance Var + alpha*Mean
        # hp["lr_sp"] = trial.suggest_float("lr_sp", 1e-2, 5e-1, log=True)       # SAP learning rate
        hp["env_num"] = trial.suggest_int("env_num", 2, 3)                      # #environments (groups)
        # partition_times impacts runtime heavily; keep default (3).

    if model == "edge_adder":
        # Candidate edges per node (compute grows with k)
        if dataset in SMALL:
            hp["edge_k"] = trial.suggest_int("edge_k", 1, 4)
        else:
            hp["edge_k"] = trial.suggest_int("edge_k", 1, 3)
        hp["lambda_dp"] = trial.suggest_float("lambda_dp", 1e-3, 1.0, log=True)
        hp["lambda_edge_l1"] = trial.suggest_float("lambda_edge_l1", 1e-5, 1e-2, log=True)

    return hp

# -----------------------------
# Optuna objective
# -----------------------------

def build_timestamp_dir(root: Path, model: str, encoder: str, dataset: str, objective: str, tag: str, study_stamp: str) -> Path:
    base = root / dataset / encoder / model / objective
    stamp = study_stamp + (f"_{tag}" if tag else "")
    return base / f"{stamp}"

def build_trial_dir(root: Path, model: str, encoder: str, dataset: str, objective: str, tag: str, study_stamp: str, trial_number: int, hp: Dict[str, Any]) -> Path:
    meta = {"m": model, "enc": encoder, "ds": dataset, "obj": objective, **hp}
    h = md5_of(meta)
    base = build_timestamp_dir(root, model, encoder, dataset, objective, tag, study_stamp)
    return base / f"trial_{trial_number:04d}_{h}"

def run_one_trial(args, device, data: FairDataset, trial: optuna.trial.Trial, seeds: List[int], study_stamp:str) -> Tuple[float, float, Path]:
    hp = suggest_hparams(trial, args.model, args.encoder, args.dataset, args.tune_scope, args.attack)
    trial.set_user_attr("hparams", hp)
    trial_dir = build_trial_dir(Path(args.log_root), args.model, args.encoder, args.dataset, args.objective, args.tag, study_stamp, trial.number, hp)
    ensure_dir(trial_dir)
    with (trial_dir / "args_trial.json").open("w") as f:
        json.dump({"hparams": hp, "objective": args.objective, "balanced_on": args.balanced_on, "w_dp": args.w_dp, "w_eo": args.w_eo}, f, indent=2)

    # Run across seeds
    for seed in seeds:
        # Prepare a per-seed args-like object
        a = copy.deepcopy(args)
        for k, v in hp.items():
            setattr(a, k, v)
        a.start_seed = seed
        a.seed_num = 1
        a.device = device
        a.cuda = torch.cuda.is_available()
        a.log_dir = str(trial_dir)
        a.seed_dir = str(trial_dir / f"seed_{seed}")
        # Ensure we are actually running the attack for this study
        if args.attack == "nifa":
            a.attack = "nifa"
        os.makedirs(a.seed_dir, exist_ok=True)

        set_seed(seed, use_cuda=a.cuda)
        if a.model == "fairinv":
            pbar = tqdm(total=args.epochs, desc=f"Seed {seed}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
            run_fairinv(a, data, pbar)
        elif a.model in ["edge_adder","edge_minmax"]:
            run_edge_adder_unified(a, data, a.seed_dir)
        else:  # vanilla
            run_vanilla(a, data, a.seed_dir)

    # Summarize
    summ = summarize_trial_dir(trial_dir, args.objective, args.balanced_on, args.w_dp, args.w_eo)
    with (trial_dir / "trial_summary.json").open("w") as f:
        json.dump(summ, f, indent=2)

    vm = summ.get("val_metric_stats", {}) or {}
    f1_mean = vm.get("f1_mean", float("-inf"))
    f1_std = vm.get("f1_std", float("inf"))
    auc_mean = vm.get("auc_mean", float("-inf"))
    auc_std = vm.get("auc_std", float("inf"))

    if args.objective == "f1_mean_minus_std" and f1_mean is not None and f1_std is not None:
        val_score_for_study = float(f1_mean) - float(f1_std)
    elif args.objective == "auc_f1_mean_minus_std" and all(v is not None for v in [f1_mean, f1_std, auc_mean, auc_std]):
        val_score_for_study = 0.5 * ((float(f1_mean) - float(f1_std)) + (float(auc_mean) - float(auc_std)))
    else:
        val_score_for_study = summ["val_mean"] if summ["val_mean"] is not None else float("-inf")

    # keep legacy attrs for debugging
    val_mean = summ["val_mean"] if summ["val_mean"] is not None else float("-inf")
    test_mean = summ["test_mean"] if summ["test_mean"] is not None else float("nan")
    trial.set_user_attr("val_mean", float(val_mean))
    trial.set_user_attr("test_mean", float(test_mean))
    trial.set_user_attr("val_f1_mean", f1_mean)
    trial.set_user_attr("val_f1_std", f1_std)
    trial.set_user_attr("val_auc_mean", auc_mean)
    trial.set_user_attr("val_auc_std", auc_std)

    return float(val_score_for_study), float(test_mean), trial_dir

# -----------------------------
# CLI & main
# -----------------------------

def make_parser():
    base = get_base_args()  # grab defaults to mirror train.py
    p = argparse.ArgumentParser(description="Optuna tuner for Fair GNNs")
    p.add_argument("--model", choices=["vanilla", "fairinv", "edge_adder"], default=base.model)
    p.add_argument("--encoder", choices=["gcn", "gat", "gin", "sage", "sgc"], default=base.encoder)
    p.add_argument("--dataset", choices=["nba", "bail", "pokec_z", "pokec_n", "german"], default=base.dataset)
    p.add_argument("--best_overall_path", type=str, default=getattr(base, "best_overall_path", ""),
                   help="Path to a JSON containing prior best victim-GNN hyperparams (will be loaded before tuning).")

    p.add_argument("--epochs", type=int, default=base.epochs)
    p.add_argument("--log_root", type=str, default="logs/optuna")
    p.add_argument("--log_interval", type=int, default=base.log_interval)

    # FairINV - SAP
    p.add_argument("--partition_times", type=int, default=base.partition_times,
                   help='the number for partitioning the sensitive attribute group.')

    # Threads
    p.add_argument("--num_threads", type=int, default=base.num_threads,
                   help="Number of CPU threads to use for BLAS/DGL/PyTorch ops.")

    # Seeds
    p.add_argument("--seeds", type=int, nargs="+", default=[base.start_seed + i for i in range(max(1, base.seed_num or 1))])
    p.add_argument("--start_seed", type=int, default=base.start_seed)
    p.add_argument("--seed_num", type=int, default=base.seed_num or 1)

    # Objective
    p.add_argument("--objective", type=str, default="auc_f1",
                   choices=["f1", "auc", "auc_f1", "balanced",
                            "f1_mean_minus_std", "auc_f1_mean_minus_std",
                            "attack_dp_eo", "attack_balanced"])
    p.add_argument("--balanced_on", choices=["auc", "f1"], default="f1")
    p.add_argument("--w_dp", type=float, default=1.0)
    p.add_argument("--w_eo", type=float, default=1.0)
    p.add_argument("--util_on", choices=["auc","f1"], default="f1")
    p.add_argument("--util_min", type=float, default=None, help="Hard utility floor for attack objectives.")
    p.add_argument("--lambda_util", type=float, default=1.0, help="Hinge penalty for attack_balanced.")

    # Attack control (we’ll keep GNN HPs fixed and only tune attack HPs)
    p.add_argument("--attack", choices=["none", "nifa"], default="none")
    p.add_argument("--tune_scope", choices=["gnn", "attack", "both"], default="attack",
                   help="What to tune: victim GNN, attack, or both. For NIFA studies use 'attack'.")

    # Optuna controls
    p.add_argument("--n_trials", type=int, default=40)
    p.add_argument("--study_name", type=str, default="auto")
    p.add_argument("--storage", type=str, default=None, help="e.g., sqlite:///optuna.db")
    p.add_argument("--sampler", type=str, choices=["tpe","random"], default="tpe")
    p.add_argument("--pruner", type=str, choices=["none","median"], default="none")
    p.add_argument("--tag", type=str, default="")

    return p

def main():
    parser = make_parser()
    args = parser.parse_args()
    args = load_best_overall_into_args(args)

    # env & device & data
    configure_threads(getattr(args, "num_threads", 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FairDataset(args.dataset, device)
    ds.load_data()

    # Build seeds list
    seeds = args.seeds if (args.seeds and len(args.seeds) > 0) else [args.start_seed + i for i in range(max(1, args.seed_num))]

    # Sampler & pruner
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True) if args.sampler == "tpe" else optuna.samplers.RandomSampler()
    pruner = None if args.pruner == "none" else optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    # Study naming
    if args.study_name == "auto":
        study_name = f"{args.model}-{args.encoder}-{args.dataset}-{args.objective}"
    else:
        study_name = args.study_name

    study_stamp = now_ts()
    out_root = build_timestamp_dir(Path(args.log_root), args.model, args.encoder, args.dataset, args.objective, args.tag, study_stamp)
    ensure_dir(out_root)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name, sampler=sampler, pruner=pruner,
        storage=args.storage, load_if_exists=True
    )

    def _objective(trial: optuna.trial.Trial):
        val_mean, test_mean, tdir = run_one_trial(args, device, ds, trial, seeds, study_stamp)
        # Report once at the end (no pruning mid-run)
        trial.set_user_attr("trial_dir", str(tdir))
        return val_mean

    study.optimize(_objective, n_trials=args.n_trials, show_progress_bar=True)

    # Save study artifacts
    best = {
        "number": study.best_trial.number,
        "value": float(study.best_value),
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
    }
    with (out_root / "best_overall.json").open("w") as f:
        json.dump(best, f, indent=2)

    try:
        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        df.to_csv(out_root / f"optuna_history_{now_ts()}.csv", index=False)
    except Exception as e:
        print("[optuna] Could not write dataframe CSV:", e)

    print(f"[optuna] Study '{study.study_name}' finished. Best value = {study.best_value:.4f}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method("spawn", force=True)
    main()
