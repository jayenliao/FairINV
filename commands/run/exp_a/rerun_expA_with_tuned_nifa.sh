#!/usr/bin/env bash
set -euo pipefail

# =========================
# Re-run Experiment A (poisoning): attack applied BEFORE training
# Uses BOTH:
#   1) victim_best_paths[ds]         (victim hyperparams)
#   2) strongest_nifa_paths[ds]      (tuned NIFA hyperparams)
# =========================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

MODEL=${MODEL:-vanilla}   # vanilla | fairinv | edge_adder | edge_minmax
ENCODER=${ENCODER:-gin}

SEED_NUM=${SEED_NUM:-5}
START_SEED=${START_SEED:-42}
EPOCHS=${EPOCHS:-1000}

RUN_LOG_ROOT=${RUN_LOG_ROOT:-logs/rerun_expA_vanilla_strongest_nifa}

datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

# Victim tuned hyperparams (your existing files)
declare -A victim_best_paths=(
  ["german"]="best_overall_json/optuna_big/vanilla/gin/german.json"
  ["bail"]="best_overall_json/optuna_big/vanilla/gin/bail.json"
  ["nba"]="best_overall_json/optuna_big/vanilla/gin/nba.json"
  ["pokec_z"]="best_overall_json/optuna_big/vanilla/gin/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_big/vanilla/gin/pokec_n.json"
)

# Strongest NIFA tuned hyperparams (FILL THESE)
# Option A: point to your Optuna NIFA study outputs (best_overall.json).
# Option B: if you copied them elsewhere, point there.
declare -A strongest_nifa_paths=(
  # مثال：["german"]="logs/optuna_nifa_expA_vanilla/german/gin/vanilla/attack_balanced/2026.../best_overall.json"
  # Fill with your actual paths:
  ["german"]="PUT_PATH_HERE"
  ["bail"]="PUT_PATH_HERE"
  ["nba"]="PUT_PATH_HERE"
  ["pokec_z"]="PUT_PATH_HERE"
  ["pokec_n"]="PUT_PATH_HERE"
)

mkdir -p "${RUN_LOG_ROOT}"

for ds in "${datasets[@]}"; do
  echo
  echo "=============================="
  echo "Re-run Exp-A | ds=${ds} | ${MODEL}/${ENCODER}"
  echo "=============================="

  vpath="${victim_best_paths[$ds]:-}"
  apath="${strongest_nifa_paths[$ds]:-}"

  if [[ -z "$vpath" || ! -f "$vpath" ]]; then
    echo "[ERROR] victim_best_paths missing or not found for ${ds}: ${vpath}"
    exit 1
  fi
  if [[ -z "$apath" || ! -f "$apath" ]]; then
    echo "[ERROR] strongest_nifa_paths missing or not found for ${ds}: ${apath}"
    exit 1
  fi

  echo "[info] victim best_overall:   $vpath"
  echo "[info] strongest NIFA best:   $apath"

  python train.py \
    --dataset "${ds}" \
    --model "${MODEL}" \
    --encoder "${ENCODER}" \
    --attack nifa \
    --attack_when train \
    --seed_num "${SEED_NUM}" \
    --start_seed "${START_SEED}" \
    --epochs "${EPOCHS}" \
    --log_dir "${RUN_LOG_ROOT}" \
    --best_overall_path "$vpath" "$apath"
done
