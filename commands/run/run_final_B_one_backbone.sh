#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash commands/final/run_final_B_one_backbone.sh gcn
#   RUN_CLEAN=0 CUDA_VISIBLE_DEVICES=0 bash commands/final/run_final_B_one_backbone.sh gcn

ENCODER="${1:-}"
if [[ -z "${ENCODER}" ]]; then
  echo "[ERROR] Missing backbone/encoder argument. Example: bash $0 gcn"
  exit 1
fi

# --------------------
# Settings (override via env)
# --------------------
DATASETS=(${DATASETS:-pokec_z pokec_n german bail nba})

START_SEED="${START_SEED:-42}"
SEED_NUM="${SEED_NUM:-10}"

DEVICE="${DEVICE:-cuda}"
NUM_THREADS="${NUM_THREADS:-2}"

EPOCHS_FINAL="${EPOCHS_FINAL:-500}"
PRETRAIN_EPOCHS_FINAL="${PRETRAIN_EPOCHS_FINAL:-500}"
EDGE_EPOCHS_FINAL="${EDGE_EPOCHS_FINAL:-500}"

RUN_CLEAN="${RUN_CLEAN:-1}"
RUN_B="${RUN_B:-1}"

# NIFA params shared (override via env)
NIFA_MODE="${NIFA_MODE:-degree}"
NIFA_T="${NIFA_T:-20}"
NIFA_THETA="${NIFA_THETA:-0.5}"
NIFA_ALPHA="${NIFA_ALPHA:-1.0}"
NIFA_BETA="${NIFA_BETA:-1.0}"
NIFA_RATIO="${NIFA_RATIO:-0.5}"
NIFA_EPOCHS="${NIFA_EPOCHS:-1000}"
NIFA_LR="${NIFA_LR:-0.001}"
NIFA_LOOPS="${NIFA_LOOPS:-50}"

CFG_ROOT="best_overall_json/optuna_big"
LOG_ROOT="${LOG_ROOT:-logs/final_B_optuna_big}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"

run_one () {
  local name="$1"
  local out="$2"
  shift 2
  mkdir -p "$(dirname "$out")"
  echo "============================================================"
  echo "${name}"
  echo "============================================================"
  python -u train.py "$@" 2>&1 | tee "$out"
}

for dataset in "${DATASETS[@]}"; do
  # ---------------------------------------------------------
  # Dataset-specific: NIFA + fairness weights (exactly your mapping)
  # ---------------------------------------------------------
  if [[ "$dataset" == "pokec_n" ]]; then
    nifa_node=87
    nifa_edge=50
  elif [[ "$dataset" == "pokec_z" ]]; then
    nifa_node=102
    nifa_edge=50
  elif [[ "$dataset" == "bail" ]]; then
    nifa_node=25
    nifa_edge=50
  elif [[ "$dataset" == "nba" ]]; then
    nifa_node=4
    nifa_edge=15
  elif [[ "$dataset" == "german" ]]; then
    nifa_node=10
    nifa_edge=50
  else
    echo "[ERROR] Unknown dataset=${dataset}. Please extend dataset->(nifa_node,nifa_edge,lambda_dp,lambda_eo) mapping."
    exit 1
  fi

  echo
  echo "============ ${dataset^^} ============="
  echo "[INFO] dataset=${dataset} encoder=${ENCODER}"
  echo "[INFO] nifa_node=${nifa_node} nifa_edge=${nifa_edge}"

  VCFG="${CFG_ROOT}/vanilla/${ENCODER}/${dataset}.json"
  ECFG="${CFG_ROOT}/edge_adder/${ENCODER}/${dataset}.json"

  if [[ ! -f "${VCFG}" ]]; then
    echo "[ERROR] Missing vanilla config: ${VCFG}"
    exit 1
  fi
  if [[ ! -f "${ECFG}" ]]; then
    echo "[ERROR] Missing edge_adder config: ${ECFG}"
    exit 1
  fi

  # --------------------
  # VANILLA
  # --------------------
  if [[ "${RUN_CLEAN}" == "1" ]]; then
    OUTDIR="${LOG_ROOT}/${dataset}/${ENCODER}/vanilla/clean/${STAMP}"
    mkdir -p "${OUTDIR}"
    cp -f "${VCFG}" "${OUTDIR}/best_overall_used.json"

    run_one \
      "[VANILLA][CLEAN] dataset=${dataset} enc=${ENCODER}" \
      "${OUTDIR}/stdout.out" \
      --model vanilla \
      --dataset "${dataset}" --encoder "${ENCODER}" \
      --device "${DEVICE}" --num_threads "${NUM_THREADS}" \
      --start_seed "${START_SEED}" --seed_num "${SEED_NUM}" \
      --epochs "${EPOCHS_FINAL}" \
      --lambda_dp 0.0 --lambda_eo 0.0 \
      --attack none \
      --best_overall_path "${VCFG}" \
      --log_dir "${OUTDIR}"
  fi

  if [[ "${RUN_B}" == "1" ]]; then
    OUTDIR="${LOG_ROOT}/${dataset}/${ENCODER}/vanilla/B_eval/${STAMP}"
    mkdir -p "${OUTDIR}"
    cp -f "${VCFG}" "${OUTDIR}/best_overall_used.json"

    run_one \
      "[VANILLA][B] dataset=${dataset} enc=${ENCODER} (attack_when=eval)" \
      "${OUTDIR}/stdout.out" \
      --model vanilla \
      --dataset "${dataset}" --encoder "${ENCODER}" \
      --device "${DEVICE}" --num_threads "${NUM_THREADS}" \
      --start_seed "${START_SEED}" --seed_num "${SEED_NUM}" \
      --epochs "${EPOCHS_FINAL}" \
      --lambda_dp 0.0 --lambda_eo 0.0 \
      --attack nifa --attack_when eval \
      --nifa_mode "${NIFA_MODE}" \
      --nifa_node "${nifa_node}" --nifa_edge "${nifa_edge}" \
      --nifa_T "${NIFA_T}" --nifa_theta "${NIFA_THETA}" \
      --nifa_alpha "${NIFA_ALPHA}" --nifa_beta "${NIFA_BETA}" --nifa_ratio "${NIFA_RATIO}" \
      --nifa_epochs "${NIFA_EPOCHS}" --nifa_lr "${NIFA_LR}" --nifa_loops "${NIFA_LOOPS}" \
      --best_overall_path "${VCFG}" \
      --log_dir "${OUTDIR}"
  fi

  # --------------------
  # EDGE_ADDER (2-stage pipeline)
  # --------------------
  if [[ "${RUN_CLEAN}" == "1" ]]; then
    OUTDIR="${LOG_ROOT}/${dataset}/${ENCODER}/edge_adder/clean/${STAMP}"
    mkdir -p "${OUTDIR}"
    cp -f "${ECFG}" "${OUTDIR}/best_overall_used.json"

    run_one \
      "[EDGE_ADDER][CLEAN] dataset=${dataset} enc=${ENCODER}" \
      "${OUTDIR}/stdout.out" \
      --model edge_adder \
      --edge_pipeline freeze_gnn_then_edge \
      --dataset "${dataset}" --encoder "${ENCODER}" \
      --device "${DEVICE}" --num_threads "${NUM_THREADS}" \
      --start_seed "${START_SEED}" --seed_num "${SEED_NUM}" \
      --epochs "${EPOCHS_FINAL}" \
      --pretrain_epochs "${PRETRAIN_EPOCHS_FINAL}" \
      --edge_epochs "${EDGE_EPOCHS_FINAL}" \
      --attack none \
      --best_overall_path "${ECFG}" \
      --log_dir "${OUTDIR}"
  fi

  if [[ "${RUN_B}" == "1" ]]; then
    OUTDIR="${LOG_ROOT}/${dataset}/${ENCODER}/edge_adder/B_eval/${STAMP}"
    mkdir -p "${OUTDIR}"
    cp -f "${ECFG}" "${OUTDIR}/best_overall_used.json"

    run_one \
      "[EDGE_ADDER][B] dataset=${dataset} enc=${ENCODER} (attack_when=eval)" \
      "${OUTDIR}/stdout.out" \
      --model edge_adder \
      --edge_pipeline freeze_gnn_then_edge \
      --dataset "${dataset}" --encoder "${ENCODER}" \
      --device "${DEVICE}" --num_threads "${NUM_THREADS}" \
      --start_seed "${START_SEED}" --seed_num "${SEED_NUM}" \
      --epochs "${EPOCHS_FINAL}" \
      --pretrain_epochs "${PRETRAIN_EPOCHS_FINAL}" \
      --edge_epochs "${EDGE_EPOCHS_FINAL}" \
      --attack nifa --attack_when eval \
      --nifa_mode "${NIFA_MODE}" \
      --nifa_node "${nifa_node}" --nifa_edge "${nifa_edge}" \
      --nifa_T "${NIFA_T}" --nifa_theta "${NIFA_THETA}" \
      --nifa_alpha "${NIFA_ALPHA}" --nifa_beta "${NIFA_BETA}" --nifa_ratio "${NIFA_RATIO}" \
      --nifa_epochs "${NIFA_EPOCHS}" --nifa_lr "${NIFA_LR}" --nifa_loops "${NIFA_LOOPS}" \
      --best_overall_path "${ECFG}" \
      --log_dir "${OUTDIR}"
  fi

done

echo
echo "✅ Done backbone=${ENCODER}"
echo "Logs: ${LOG_ROOT} (organized by dataset/encoder/model)"
