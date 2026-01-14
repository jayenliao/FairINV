#!/usr/bin/env bash
set -euo pipefail

# =========================
# Re-run Experiment B (eval-only NIFA) for EdgeAdder using FIXED tuned HP paths
# - Victim: EdgeAdder (clean training)
# - Attack: NIFA applied ONLY at evaluation (attack_when=eval)
# - Keep 2-stage (freeze_gnn_then_edge) + alternating training
# - Load BOTH victim tuned HPs and tuned NIFA HPs via --best_overall_path
#   (victim first, NIFA second)
# =========================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

MODEL=${MODEL:-edge_adder}
ENCODER=${ENCODER:-sage}

SEED_NUM=${SEED_NUM:-10}
START_SEED=${START_SEED:-42}
EPOCHS=${EPOCHS:-1000}

# --- 2-stage + alternating training settings ---
EDGE_PIPELINE=${EDGE_PIPELINE:-freeze_gnn_then_edge}
ALT_ROUNDS=${ALT_ROUNDS:-10}
ALT_EDGE_EPOCHS=${ALT_EDGE_EPOCHS:-20}
ALT_GNN_EPOCHS=${ALT_GNN_EPOCHS:-20}
ALT_GNN_LR=${ALT_GNN_LR:-}  # optional

datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

RUN_LOG_ROOT=${RUN_LOG_ROOT:-logs/rerun_expB_edge_adder_tunedEA_tunedNIFA_fixed}
mkdir -p "${RUN_LOG_ROOT}"

# -------------------------
# Victim tuned hyperparams (EdgeAdder clean)
# -------------------------
declare -A victim_best_paths=(
  ["german"]="best_overall_json/optuna_big/edge_adder/sage/german.json"
  ["bail"]="best_overall_json/optuna_big/edge_adder/sage/bail.json"
  ["nba"]="best_overall_json/optuna_big/edge_adder/sage/nba.json"
  ["pokec_z"]="best_overall_json/optuna_big/edge_adder/sage/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_big/edge_adder/sage/pokec_n.json"
)

# -------------------------
# Tuned NIFA hyperparams (FIXED PATHS — fill these in)
# IMPORTANT: These should be the best_overall.json (attack study) tuned for Exp-B
# under the SAME encoder/model you are rerunning (sage/edge_adder).
# -------------------------
declare -A strongest_nifa_paths=(
  ["german"]="best_overall_json/optuna_nifa_expB/edge_adder_expB/sage/german.json"
  ["bail"]="best_overall_json/optuna_nifa_expB/edge_adder_expB/sage/bail.json"
  ["nba"]="best_overall_json/optuna_nifa_expB/edge_adder_expB/sage/nba.json"
  ["pokec_z"]="best_overall_json/optuna_nifa_expB/edge_adder_expB/sage/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_nifa_expB/edge_adder_expB/sage/pokec_n.json"
)

for ds in "${datasets[@]}"; do
  echo
  echo "===================================================="
  echo "Re-run Exp-B (fixed paths) | ds=${ds} | ${MODEL}/${ENCODER}"
  echo "edge_pipeline=${EDGE_PIPELINE}, alt_rounds=${ALT_ROUNDS}"
  echo "===================================================="

  vpath="${victim_best_paths[$ds]:-}"
  apath="${strongest_nifa_paths[$ds]:-}"

  if [[ -z "${vpath}" || ! -f "${vpath}" ]]; then
    echo "[ERROR] victim best_overall missing/not found for ${ds}: ${vpath}"
    exit 1
  fi
  if [[ -z "${apath}" || ! -f "${apath}" ]]; then
    echo "[ERROR] tuned NIFA best_overall missing/not found for ${ds}: ${apath}"
    exit 1
  fi

  echo "[info] victim best_overall: ${vpath}"
  echo "[info] tuned  NIFA best_overall: ${apath}"

  cmd=(python train.py
    --dataset "${ds}"
    --model "${MODEL}"
    --encoder "${ENCODER}"
    --num_threads "${OMP_NUM_THREADS}"
    --attack nifa
    --attack_when eval
    --seed_num "${SEED_NUM}"
    --start_seed "${START_SEED}"
    --epochs "${EPOCHS}"
    --log_dir "${RUN_LOG_ROOT}"
    --edge_pipeline "${EDGE_PIPELINE}"
    --alt_rounds "${ALT_ROUNDS}"
    --alt_edge_epochs "${ALT_EDGE_EPOCHS}"
    --alt_gnn_epochs "${ALT_GNN_EPOCHS}"
    --best_overall_path "${vpath}" "${apath}"
  )

  if [[ -n "${ALT_GNN_LR}" ]]; then
    cmd+=(--alt_gnn_lr "${ALT_GNN_LR}")
  fi

  echo "[info] Running: ${cmd[*]}"
  "${cmd[@]}"
done
