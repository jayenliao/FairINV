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

import torch
import optuna

from args import get_parser
from data import FairDataset
from utils import set_seed
from utils import configure_threads
# training entrypoints and snapshot helpers
from train import run_vanilla, run_fairinv, run_edge_adder_unified
from train import load_best_overall_into_args, snapshot_clean_data, restore_from_snapshot
# NIFA bridge
from nifa_bridge import apply_nifa_attack

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

    # Strong checks for attack objectives — do not silently degrade to "balanced"
    def _need(keys: list[str]):
        for k in keys:
            if k not in row or row[k] is None:
                raise KeyError(f"objective_value: missing metric '{k}' in row for objective='{objective}'")

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

    # balanced (utility minus fairness penalties)
    m = fetch_metric(row, "auc") if balanced_on == "auc" else fetch_metric(row, "f1")
    dp, eo = fetch_metric(row, "dp"), fetch_metric(row, "eo")
    if m is None or dp is None or eo is None:
        return float("-inf")

    if objective == "balanced":
        return float(m) - w_dp * float(dp) - w_eo * float(eo)
    if objective == "auc_f1_balanced":
        v_auc = fetch_metric(row, "auc")
        v_f1 = fetch_metric(row, "f1")
        if v_auc is None or v_f1 is None:
            return float("-inf")
        score = 0.5 * (float(v_auc) + float(v_f1)) - w_dp * float(dp) - w_eo * float(eo)
        return score
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

def summarize_trial_dir(
    trial_dir: Path, objective: str, balanced_on: str, w_dp: float, w_eo: float,
    util_on: str = "f1", util_min: float | None = None, lambda_util: float = 1.0
) -> Dict[str, Any]:
    per_seed = []
    for sd in sorted([p for p in trial_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        rows = read_jsonl(sd / "metrics.jsonl")
        val_rows = [r for r in rows if r.get("split") == "val"]
        test_rows = [r for r in rows if r.get("split") == "test"]
        # best val row by objective
        best = None
        best_val = float("-inf")
        for r in val_rows:
            v = objective_value(r, objective, balanced_on, w_dp, w_eo, util_on, util_min, lambda_util)
            if v > best_val:
                best_val = v
                best = r
        test_row = test_rows[-1] if len(test_rows) > 0 else {}
        per_seed.append({
            "seed": int(sd.name.split("_")[-1]),
            "best_val_score": best_val,
            "best_val_row": best,
            "test_row": test_row,
            "test_score": objective_value(test_row, objective, balanced_on, w_dp, w_eo, util_on, util_min, lambda_util),
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
    dp_vals, eo_vals  = [], []
    for s in per_seed:
        r = s.get("best_val_row") or {}
        v_f1 = fetch_metric(r, "f1")
        v_auc = fetch_metric(r, "auc")
        v_dp = fetch_metric(r, "dp")
        v_eo = fetch_metric(r, "eo")
        if v_f1 is not None:
            f1_vals.append(float(v_f1))
        if v_auc is not None:
            auc_vals.append(float(v_auc))
        if v_dp is not None:
            dp_vals.append(float(v_dp))
        if v_eo is not None:
            eo_vals.append(float(v_eo))

    return {
        "val_mean": val_mean,
        "test_mean": test_mean,
        "per_seed": per_seed,
        "val_metric_stats": {
            "f1_mean": stats.mean(f1_vals) if f1_vals else None,
            "f1_std": stats.pstdev(f1_vals) if len(f1_vals) > 1 else 0.0,
            "auc_mean": stats.mean(auc_vals) if auc_vals else None,
            "auc_std": stats.pstdev(auc_vals) if len(auc_vals) > 1 else 0.0,
            "dp_mean": stats.mean(dp_vals) if dp_vals else None,
            "dp_std": stats.pstdev(dp_vals) if len(dp_vals) > 1 else 0.0,
            "eo_mean": stats.mean(eo_vals) if eo_vals else None,
            "eo_std": stats.pstdev(eo_vals) if len(eo_vals) > 1 else 0.0,
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

def suggest_hparams(
    trial: optuna.trial.Trial,
    model: str,
    encoder: str,
    dataset: str,
    tune_scope: str = "gnn",
    attack: str = "none",
    tune_subset: set[str] | None = None,
) -> Dict[str, Any]:
    """Return a dict of suggested hyperparams for this (model, encoder, dataset).

    If tune_subset is provided, Optuna will only suggest parameters whose names are
    contained in the subset; all other hyperparameters remain fixed (typically from
    CLI defaults and/or --best_overall_path overrides).
    """
    hp: Dict[str, Any] = {}

    def want(name: str) -> bool:
        return (tune_subset is None) or (name in tune_subset)

    # If only tuning attack, return NIFA hparams directly
    if tune_scope in {"attack", "both"} and attack == "nifa":
        if tune_subset is None:
            hp.update(_suggest_nifa_hparams(trial, dataset))
        else:
            # Tune only selected NIFA knobs
            if want("nifa_node"):
                hp["nifa_node"] = trial.suggest_int("nifa_node", 1, 200, step=1)
            if want("nifa_edge"):
                hp["nifa_edge"] = trial.suggest_int("nifa_edge", 1, 200, step=1)
            if want("nifa_loops"):
                hp["nifa_loops"] = trial.suggest_int("nifa_loops", 1, 10, step=1)
        if tune_scope == "attack":
            return hp

    # Common spaces: learning rate, weight decay, hidden size, and fairness lambdas.
    # If tune_subset is provided, we allow tuning of any included keys even if
    # tune_scope would normally exclude them.
    _scope_allows_gnn = (tune_scope in {"gnn", "both"}) or (
        tune_subset is not None and any(
            k in tune_subset
            for k in [
                "lr",
                "weight_decay",
                "hid_dim",
                "lambda_dp",
                "lambda_eo",
                "pretrain_lambda_dp",
                "pretrain_lambda_eo",
            ]
        )
    )
    if _scope_allows_gnn:
        if dataset in SMALL:
            if want("lr"):
                hp["lr"] = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
            if want("weight_decay"):
                hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            if want("hid_dim"):
                hp["hid_dim"] = trial.suggest_categorical("hid_dim", [16, 32, 64])
        else:  # LARGE
            if want("lr"):
                hp["lr"] = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
            if want("weight_decay"):
                hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
            if want("hid_dim"):
                hp["hid_dim"] = trial.suggest_categorical("hid_dim", [32, 64, 128])

        # Fairness lambdas: allow a "zero" option unless vanilla.
        if model == "vanilla":
            if want("lambda_dp"):
                hp["lambda_dp"] = 0.0
            if want("lambda_eo"):
                hp["lambda_eo"] = 0.0
        else:
            if want("lambda_dp"):
                use_zero_dp = trial.suggest_categorical("use_zero_dp", [True, False])
                hp["lambda_dp"] = 0.0 if use_zero_dp else trial.suggest_float("lambda_dp", 1e-4, 100.0, log=True)
            if want("lambda_eo"):
                # If lambda_dp isn't tuned, we don't force any coupling here.
                use_zero_eo = trial.suggest_categorical("use_zero_eo", [True, False])
                hp["lambda_eo"] = 0.0 if use_zero_eo else trial.suggest_float("lambda_eo", 1e-4, 100.0, log=True)

        # Optional: separate fairness strength for stage-1 pretraining (pipeline-A).
        # This lets you keep stage-1 clean (0) while tuning stage-3 lambdas, or vice versa.
        if want("pretrain_lambda_dp"):
            use_zero_pre_dp = trial.suggest_categorical("use_zero_pretrain_dp", [True, False])
            hp["pretrain_lambda_dp"] = 0.0 if use_zero_pre_dp else trial.suggest_float("pretrain_lambda_dp", 1e-4, 10.0, log=True)
        if want("pretrain_lambda_eo"):
            use_zero_pre_eo = trial.suggest_categorical("use_zero_pretrain_eo", [True, False])
            hp["pretrain_lambda_eo"] = 0.0 if use_zero_pre_eo else trial.suggest_float("pretrain_lambda_eo", 1e-4, 10.0, log=True)

    if want("dropout"):
        hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.7)
    # layer_num: used by GCN/GAT; SGC ignores beyond 1; GIN/SAGE custom modules ignore layer_num internally.
    # if model == "fairinv":
    #     if encoder in ["gcn", "gin", "sgc"]:
    #         hp["layer_num"] = 1
    #     elif encoder in ["sage"]:
    #         hp["layer_num"] = 1
    #     else: # gat
    #         hp["layer_num"] = 1
    # else:
    #     if encoder in {"gcn", "gat"}:
    #         hp["layer_num"] = 1
    #     elif encoder == "sgc":
    #         hp["layer_num"] = 1  # ConstructModel stacks a single SGConv
    #     else: # sage, gin
    #         hp["layer_num"] = 1

    # Model-specific
    if model == "fairinv":
        if want("alpha"):
            hp["alpha"] = trial.suggest_categorical("alpha", [0.001, 0.01, 0.1, 0.5, 1, 10, 100])
        if want("lr_sp"):
            hp["lr_sp"] = trial.suggest_categorical("lr_sp", [0.01, 0.05, 0.1, 0.5])
        # hp["alpha"] = trial.suggest_float("alpha", 1e-1, 1e+1, log=True)       # balance Var + alpha*Mean
        # hp["lr_sp"] = trial.suggest_float("lr_sp", 1e-2, 5e-1, log=True)       # SAP learning rate
        if want("env_num"):
            hp["env_num"] = trial.suggest_int("env_num", 2, 3)                      # #environments (groups)
        # partition_times impacts runtime heavily; keep default (3).

    if model == "edge_adder" and tune_scope in {"gnn", "both", "edge_adder"}:
        # Candidate edges per node (compute grows with k)
        if dataset in SMALL:
            if want("edge_k"):
                hp["edge_k"] = trial.suggest_int("edge_k", 1, 4)
        else:
            if want("edge_k"):
                hp["edge_k"] = trial.suggest_int("edge_k", 1, 3)
        if want("lambda_edge_l1"):
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
    tune_subset = None
    if getattr(args, "tune_subset", None):
        tune_subset = {s.strip() for s in str(args.tune_subset).split(",") if s.strip()}
    hp = suggest_hparams(
        trial,
        args.model,
        args.encoder,
        args.dataset,
        args.tune_scope,
        args.attack,
        tune_subset=tune_subset,
    )
    trial.set_user_attr("hparams", hp)
    trial_dir = build_trial_dir(Path(args.log_root), args.model, args.encoder, args.dataset, args.objective, args.tag, study_stamp, trial.number, hp)
    ensure_dir(trial_dir)
    with (trial_dir / "args_trial.json").open("w") as f:
        json.dump({
            "hparams": hp,
            "objective": args.objective,
            "balanced_on": args.balanced_on,
            "w_dp": args.w_dp,
            "w_eo": args.w_eo,
            "util_on": getattr(args, "util_on", "f1"),
            "util_min": getattr(args, "util_min", None),
            "lambda_util": getattr(args, "lambda_util", 1.0),
        }, f, indent=2)

    # Take a clean snapshot once; restore per seed to avoid cumulative mutations
    clean_snap = snapshot_clean_data(data)

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

        # fresh data per seed
        data_seed = restore_from_snapshot(clean_snap, device)

        # apply attack before training (if attack_when includes 'train')
        attack_when = getattr(a, 'attack_when', 'train')
        if getattr(a, 'attack', 'none') == 'nifa' and attack_when in ('train', 'both'):
            # IMPORTANT: apply the attack to this per-seed fresh copy (avoid mutating the shared original)
            data_seed = apply_nifa_attack(a, data_seed)

        # dispatch trainer on (possibly) attacked graph
        if a.model == "fairinv":
            pbar = tqdm(total=args.epochs, desc=f"Seed {seed}", unit="epoch", bar_format="{l_bar}{bar:30}{r_bar}")
            run_fairinv(a, data_seed, pbar)
        elif a.model in ["edge_adder","edge_minmax"]:
            run_edge_adder_unified(a, data_seed, a.seed_dir)
        else:  # vanilla
            run_vanilla(a, data_seed, a.seed_dir)


    # Summarize
    summ = summarize_trial_dir(
        trial_dir, args.objective, args.balanced_on, args.w_dp, args.w_eo,
        getattr(args, "util_on", "f1"),
        getattr(args, "util_min", 0.55),
        getattr(args, "lambda_util", 1.0),
    )

    # A small debug snapshot to inspect chosen epoch/metrics per seed
    try:
        dbg = {
            "objective": args.objective,
            "balanced_on": args.balanced_on,
            "util_on": getattr(args, "util_on", "f1"),
            "util_min": getattr(args, "util_min", None),
            "lambda_util": getattr(args, "lambda_util", 1.0),
            "per_seed": [
                {
                    "seed": s["seed"],
                    "val_score": s["best_val_score"],
                    "chosen_val_epoch": (s["best_val_row"] or {}).get("epoch"),
                    "f1": (s["best_val_row"] or {}).get("f1"),
                    "auc": (s["best_val_row"] or {}).get("auc"),
                    "dp": (s["best_val_row"] or {}).get("dp"),
                    "eo": (s["best_val_row"] or {}).get("eo"),
                } for s in summ.get("per_seed", [])
            ],
        }
        with (trial_dir / "chosen_val_debug.json").open("w") as f:
            json.dump(dbg, f, indent=2)
    except Exception as _e:
        pass

    with (trial_dir / "trial_summary.json").open("w") as f:
        json.dump(summ, f, indent=2)

    vm = summ.get("val_metric_stats", {}) or {}
    f1_mean  = vm.get("f1_mean", float("-inf"))
    f1_std   = vm.get("f1_std", float("inf"))
    auc_mean = vm.get("auc_mean", float("-inf"))
    auc_std  = vm.get("auc_std", float("inf"))
    dp_mean  = vm.get("dp_mean", float("-inf"))
    dp_std   = vm.get("dp_std", float("inf"))
    eo_mean  = vm.get("eo_mean", float("-inf"))
    eo_std   = vm.get("eo_std", float("inf"))

    if args.objective == "f1_mean_minus_std" and f1_mean is not None and f1_std is not None:
        val_score_for_study = float(f1_mean) - float(f1_std)
    elif args.objective == "auc_f1_mean_minus_std" and all(v is not None for v in [f1_mean, f1_std, auc_mean, auc_std]):
        val_score_for_study = 0.5 * ((float(f1_mean) - float(f1_std)) + (float(auc_mean) - float(auc_std)))
    elif args.objective == "auc_f1_balanced":
        val_score_for_study = summ["val_mean"] - auc_std - f1_std - dp_std - eo_std
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
    trial.set_user_attr("val_dp_mean", dp_mean)
    trial.set_user_attr("val_dp_std", dp_std)
    trial.set_user_attr("val_eo_mean", eo_mean)
    trial.set_user_attr("val_eo_std", eo_std)

    return float(val_score_for_study), float(test_mean), trial_dir

# -----------------------------
# CLI & main
# -----------------------------

def make_parser():
    # base = get_base_args()  # grab defaults to mirror train.py
    p = get_parser()
    # p = argparse.ArgumentParser(description="Optuna tuner for Fair GNNs")
    # p.add_argument("--model", choices=["vanilla", "fairinv", "edge_adder"], default=base.model)
    # p.add_argument("--encoder", choices=["gcn", "gat", "gin", "sage", "sgc"], default=base.encoder)
    # p.add_argument("--dataset", choices=["nba", "bail", "pokec_z", "pokec_n", "german"], default=base.dataset)
    # p.add_argument("--best_overall_path", type=str, default=getattr(base, "best_overall_path", ""),
    #                help="Path to a JSON containing prior best victim-GNN hyperparams (will be loaded before tuning).")

    # p.add_argument("--epochs", type=int, default=base.epochs)
    # p.add_argument("--log_interval", type=int, default=base.log_interval)
    p.add_argument("--log_root", type=str, default="logs/optuna")

    # FairINV - SAP
    # p.add_argument("--partition_times", type=int, default=base.partition_times,
    #                help='the number for partitioning the sensitive attribute group.')

    # Threads
    # p.add_argument("--num_threads", type=int, default=base.num_threads,
    #                help="Number of CPU threads to use for BLAS/DGL/PyTorch ops.")

    # Seeds
    # p.add_argument("--seeds", type=int, nargs="+", default=[base.start_seed + i for i in range(max(1, base.seed_num or 1))])
    # p.add_argument("--start_seed", type=int, default=base.start_seed)
    # p.add_argument("--seed_num", type=int, default=base.seed_num or 1)

    # Objective
    p.add_argument("--objective", type=str, default="auc_f1",
                   choices=["f1", "auc", "auc_f1", "balanced", "auc_f1_balanced",
                            "f1_mean_minus_std", "auc_f1_mean_minus_std",
                            "attack_dp_eo", "attack_balanced"])
    p.add_argument("--balanced_on", choices=["auc", "f1"], default="f1")
    p.add_argument("--w_dp", type=float, default=1.0)
    p.add_argument("--w_eo", type=float, default=1.0)
    p.add_argument("--util_on", choices=["auc","f1"], default="f1")
    p.add_argument("--util_min", type=float, default=None, help="Hard utility floor for attack objectives.")
    p.add_argument("--lambda_util", type=float, default=1.0, help="Hinge penalty for attack_balanced.")

    # Attack control (we’ll keep GNN HPs fixed and only tune attack HPs)
    # p.add_argument("--attack", choices=["none", "nifa"], default="none")
    p.add_argument("--tune_scope", choices=["gnn", "edge_adder", "attack", "both"], default="gnn",
                   help="What to tune: victim GNN, attack, or both. For NIFA studies use 'attack'.")

    p.add_argument(
        "--tune_subset",
        type=str,
        default=None,
        help=(
            "Comma-separated list of hyperparameter names to tune. If provided, Optuna will only sample "
            "these keys and keep all other hyperparameters fixed (typically from CLI defaults and/or --best_overall_path). "
            "Example for pipeline-A: 'lambda_dp,lambda_eo,lambda_edge_l1,edge_k'."
        ),
    )

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
    seeds = list(range(args.start_seed, args.start_seed + args.seed_num)) # if (not args.seeds or len(args.seeds) == 0) else args.seeds

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
