#!/usr/bin/env bash
set -euo pipefail

# =========================
# Re-run Experiment B (eval-only attack) using tuned hyperparameters
# - attack applied ONLY at evaluation: --attack_when eval
# - load BOTH victim HPs and tuned NIFA HPs via --best_overall_path (victim first, nifa second)
# =========================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Victim config (must match what you tuned against)
MODEL=${MODEL:-edge_adder}     # vanilla | fairinv | edge_adder | edge_minmax
ENCODER=${ENCODER:-sgc}     # gcn | gat | gin | sage | sgc

# Seeds / training budget for re-run
SEED_NUM=${SEED_NUM:-10}
START_SEED=${START_SEED:-42}
EPOCHS=${EPOCHS:-1000}

# Datasets
datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

# Where to write rerun logs
RUN_LOG_ROOT=${RUN_LOG_ROOT:-logs/rerun_expB_tuned_nifa}
mkdir -p "${RUN_LOG_ROOT}"

# -------------------------
# Victim tuned hyperparams (you already have these)
# -------------------------
declare -A victim_best_paths=(
  ["german"]="best_overall_json/optuna_big/edge_adder/sgc/german.json"
  ["bail"]="best_overall_json/optuna_big/edge_adder/sgc/bail.json"
  ["nba"]="best_overall_json/optuna_big/edge_adder/sgc/nba.json"
  ["pokec_z"]="best_overall_json/optuna_big/edge_adder/sgc/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_big/edge_adder/sgc/pokec_n.json"
)

# -------------------------
# NIFA tuned hyperparams (from Exp-B Optuna tuning logs)
# This script auto-finds the latest best_overall.json under:
#   NIFA_TUNE_LOG_ROOT/<ds>/<encoder>/<model>/<objective>/**/best_overall.json
# -------------------------
NIFA_TUNE_LOG_ROOT=${NIFA_TUNE_LOG_ROOT:-logs/optuna_nifa_expB}
NIFA_TUNE_OBJECTIVE=${NIFA_TUNE_OBJECTIVE:-attack_balanced}

for ds in "${datasets[@]}"; do
  echo
  echo "===================================================="
  echo "Re-run Exp-B (eval-only) | ds=${ds} | ${MODEL}/${ENCODER}"
  echo "===================================================="

  vpath="${victim_best_paths[$ds]:-}"
  if [[ -z "${vpath}" || ! -f "${vpath}" ]]; then
    echo "[ERROR] victim best_overall missing/not found for ${ds}: ${vpath}"
    exit 1
  fi

  base="${NIFA_TUNE_LOG_ROOT}/${ds}/${ENCODER}/${MODEL}/${NIFA_TUNE_OBJECTIVE}"
  if [[ ! -d "${base}" ]]; then
    echo "[ERROR] NIFA tune base dir not found: ${base}"
    echo "        (set NIFA_TUNE_LOG_ROOT / NIFA_TUNE_OBJECTIVE accordingly)"
    exit 1
  fi

  # pick the latest best_overall.json (sorted path order; works if your run dirs are timestamped)
  npath="$(find "${base}" -name best_overall.json | sort | tail -n 1)"
  if [[ -z "${npath}" || ! -f "${npath}" ]]; then
    echo "[ERROR] Could not find tuned NIFA best_overall.json under: ${base}"
    exit 1
  fi

  echo "[info] victim best_overall: ${vpath}"
  echo "[info] tuned  NIFA best_overall: ${npath}"

  # IMPORTANT: victim first, NIFA second (later overrides earlier if any overlap)
  python train.py \
    --dataset "${ds}" \
    --model "${MODEL}" \
    --encoder "${ENCODER}" \
    --num_threads "${OMP_NUM_THREADS}" \
    --attack nifa \
    --attack_when eval \
    --seed_num "${SEED_NUM}" \
    --start_seed "${START_SEED}" \
    --epochs "${EPOCHS}" \
    --log_dir "${RUN_LOG_ROOT}" \
    --best_overall_path "${vpath}" "${npath}"
done
