#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Exp A rerun (poison -> defense): Clean -> NIFA poison -> train EdgeAdder on poisoned graph
# Compare:
#   (A) NIFA HPs tuned with --attack_when eval  (but applied with --attack_when train)
#   (B) NIFA HPs tuned with --attack_when train (poisoning-tuned)
#
# Defense: EdgeAdder with 2-stage + alternating training.
# HP loading: --best_overall_path EA_BEST NIFA_BEST (EA first, NIFA second)
#
# Usage example:
#  CUDA_VISIBLE_DEVICES=0 ENCODER=gcn bash commands/run/rerun_expA_compare_evalHP_vs_trainHP.sh
# ============================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

MODEL=${MODEL:-edge_adder}
ENCODER=${ENCODER:-gcn}

SEED_NUM=${SEED_NUM:-10}
START_SEED=${START_SEED:-42}
EPOCHS=${EPOCHS:-1000}

# 2-stage + alternating training
EDGE_PIPELINE=${EDGE_PIPELINE:-freeze_gnn_then_edge}
ALT_ROUNDS=${ALT_ROUNDS:-10}
ALT_EDGE_EPOCHS=${ALT_EDGE_EPOCHS:-20}
ALT_GNN_EPOCHS=${ALT_GNN_EPOCHS:-20}
ALT_GNN_LR=${ALT_GNN_LR:-}  # optional

datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

# ---------- Tuned EdgeAdder (clean) ----------
declare -A ea_best_paths=(
  ["german"]="best_overall_json/optuna_big/edge_adder/${ENCODER}/german.json"
  ["bail"]="best_overall_json/optuna_big/edge_adder/${ENCODER}/bail.json"
  ["nba"]="best_overall_json/optuna_big/edge_adder/${ENCODER}/nba.json"
  ["pokec_z"]="best_overall_json/optuna_big/edge_adder/${ENCODER}/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_big/edge_adder/${ENCODER}/pokec_n.json"
)

# ---------- NIFA HPs (EVAL-tuned) ----------
# Fill these with your stable copied paths (portable across machines)
declare -A nifa_eval_hp_paths=(
  ["german"]="best_overall_json/optuna_nifa_expB/vanilla_expB/${ENCODER}/german.json"
  ["bail"]="best_overall_json/optuna_nifa_expB/vanilla_expB/${ENCODER}/bail.json"
  ["nba"]="best_overall_json/optuna_nifa_expB/vanilla_expB/${ENCODER}/nba.json"
  ["pokec_z"]="best_overall_json/optuna_nifa_expB/vanilla_expB/${ENCODER}/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_nifa_expB/vanilla_expB/${ENCODER}/pokec_n.json"
)

# ---------- NIFA HPs (TRAIN-tuned) ----------
# These do not exist yet; after you tune (script below), copy to these stable paths.
declare -A nifa_train_hp_paths=(
  ["german"]="best_overall_json/optuna_nifa_expA/vanilla_expA/${ENCODER}/german.json"
  ["bail"]="best_overall_json/optuna_nifa_expA/vanilla_expA/${ENCODER}/bail.json"
  ["nba"]="best_overall_json/optuna_nifa_expA/vanilla_expA/${ENCODER}/nba.json"
  ["pokec_z"]="best_overall_json/optuna_nifa_expA/vanilla_expA/${ENCODER}/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_nifa_expA/vanilla_expA/${ENCODER}/pokec_n.json"
)

RUN_ROOT=${RUN_ROOT:-logs/expA_compare}
mkdir -p "${RUN_ROOT}"

run_one () {
  local ds="$1"
  local label="$2"
  local nifa_path="$3"

  local ea_path="${ea_best_paths[$ds]}"
  if [[ ! -f "${ea_path}" ]]; then
    echo "[ERROR] EA best_overall not found: ${ea_path}"
    exit 1
  fi
  if [[ ! -f "${nifa_path}" ]]; then
    echo "[ERROR] NIFA best_overall not found: ${nifa_path}"
    exit 1
  fi

  local out_dir="${RUN_ROOT}/${label}"
  mkdir -p "${out_dir}"

  echo
  echo "=============================="
  echo "ExpA rerun | ${label} | ds=${ds} | ${MODEL}/${ENCODER}"
  echo "EA=${ea_path}"
  echo "NIFA=${nifa_path}"
  echo "=============================="

  cmd=(python train.py
    --dataset "${ds}"
    --model "${MODEL}"
    --encoder "${ENCODER}"
    --num_threads "${OMP_NUM_THREADS}"
    --attack nifa
    --attack_when train
    --seed_num "${SEED_NUM}"
    --start_seed "${START_SEED}"
    --epochs "${EPOCHS}"
    --log_dir "${out_dir}"
    --edge_pipeline "${EDGE_PIPELINE}"
    --alt_rounds "${ALT_ROUNDS}"
    --alt_edge_epochs "${ALT_EDGE_EPOCHS}"
    --alt_gnn_epochs "${ALT_GNN_EPOCHS}"
    --best_overall_path "${ea_path}" "${nifa_path}"
  )
  if [[ -n "${ALT_GNN_LR}" ]]; then
    cmd+=(--alt_gnn_lr "${ALT_GNN_LR}")
  fi

  echo "[info] Running: ${cmd[*]}"
  "${cmd[@]}"
}

for ds in "${datasets[@]}"; do
  # run_one "${ds}" "nifa_evalHP_applied_as_train"  "${nifa_eval_hp_paths[$ds]}"
  run_one "${ds}" "nifa_trainHP"                  "${nifa_train_hp_paths[$ds]}"
done
