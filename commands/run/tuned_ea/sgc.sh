#!/usr/bin/env bash
set -euo pipefail

# =========================
# Re-run CLEAN EdgeAdder (no NIFA) using tuned victim hyperparameters
# with 2-stage pipeline (freeze_gnn_then_edge) + alternative training.
# =========================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

MODEL=${MODEL:-edge_adder}     # vanilla | fairinv | edge_adder | edge_minmax
ENCODER=${ENCODER:-sgc}        # gcn | gat | gin | sage | sgc
SEED_NUM=${SEED_NUM:-10}
START_SEED=${START_SEED:-42}
EPOCHS=${EPOCHS:-1000}

# --- 2-stage + alternative training knobs ---
# NOTE: If your best_overall.json already contains these keys, the loaded params may override CLI.
EDGE_PIPELINE=${EDGE_PIPELINE:-freeze_gnn_then_edge}
ALT_ROUNDS=${ALT_ROUNDS:-10}
ALT_EDGE_EPOCHS=${ALT_EDGE_EPOCHS:-20}
ALT_GNN_EPOCHS=${ALT_GNN_EPOCHS:-20}
ALT_GNN_LR=${ALT_GNN_LR:-}     # optional; leave empty to use default lr

datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

RUN_LOG_ROOT=${RUN_LOG_ROOT:-logs/rerun_clean_edge_adder_alt}
mkdir -p "${RUN_LOG_ROOT}"

# Victim tuned hyperparams (EdgeAdder clean tuning)
declare -A victim_best_paths=(
  ["german"]="best_overall_json/optuna_big/edge_adder/sgc/german.json"
  ["bail"]="best_overall_json/optuna_big/edge_adder/sgc/bail.json"
  ["nba"]="best_overall_json/optuna_big/edge_adder/sgc/nba.json"
  ["pokec_z"]="best_overall_json/optuna_big/edge_adder/sgc/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_big/edge_adder/sgc/pokec_n.json"
)

for ds in "${datasets[@]}"; do
  echo
  echo "===================================================="
  echo "Re-run CLEAN EdgeAdder | ds=${ds} | ${MODEL}/${ENCODER}"
  echo "edge_pipeline=${EDGE_PIPELINE}, alt_rounds=${ALT_ROUNDS}"
  echo "===================================================="

  vpath="${victim_best_paths[$ds]:-}"
  if [[ -z "${vpath}" || ! -f "${vpath}" ]]; then
    echo "[ERROR] victim best_overall missing/not found for ${ds}: ${vpath}"
    exit 1
  fi
  echo "[info] victim best_overall: ${vpath}"

  cmd=(python train.py
    --dataset "${ds}"
    --model "${MODEL}"
    --encoder "${ENCODER}"
    --num_threads "${OMP_NUM_THREADS}"
    --attack none
    --seed_num "${SEED_NUM}"
    --start_seed "${START_SEED}"
    --epochs "${EPOCHS}"
    --log_dir "${RUN_LOG_ROOT}"
    --edge_pipeline "${EDGE_PIPELINE}"
    --alt_rounds "${ALT_ROUNDS}"
    --alt_edge_epochs "${ALT_EDGE_EPOCHS}"
    --alt_gnn_epochs "${ALT_GNN_EPOCHS}"
    --best_overall_path "${vpath}"
  )

  # only add alt_gnn_lr if provided
  if [[ -n "${ALT_GNN_LR}" ]]; then
    cmd+=(--alt_gnn_lr "${ALT_GNN_LR}")
  fi

  echo "[info] Running: ${cmd[*]}"
  "${cmd[@]}"
done
