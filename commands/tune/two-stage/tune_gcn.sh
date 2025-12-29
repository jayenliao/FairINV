#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ENCODER="gcn"
DATASETS=(pokec_z pokec_n bail german nba)

LOG_ROOT="${LOG_ROOT:-logs/optuna_big}"
TAG_PREFIX="${TAG_PREFIX:-big_clean}"
START_SEED="${START_SEED:-0}"
TUNE_SEEDS="${TUNE_SEEDS:-10}"

N_TRIALS_VANILLA="${N_TRIALS_VANILLA:-32}"
N_TRIALS_EDGE="${N_TRIALS_EDGE:-32}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-500}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-500}"
EDGE_EPOCHS="${EDGE_EPOCHS:-500}"

W_DP="${W_DP:-1.0}"
W_EO="${W_EO:-1.0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG_PREFIX}_${ENCODER}_${STAMP}"

mkdir -p "${LOG_ROOT}"

latest_best () {
  local ds="$1" enc="$2" model="$3" obj="$4"
  ls -t "${LOG_ROOT}/${ds}/${enc}/${model}/${obj}"/*/best_overall.json 2>/dev/null | head -n 1 || true
}

echo "[INFO] encoder=${ENCODER}, datasets=${DATASETS[*]}, log_root=${LOG_ROOT}, tag=${TAG}"
echo "[WARN] If you need PURE vanilla (lambda_dp=lambda_eo=0), ensure your tuner does NOT tune them for model=vanilla."

for ds in "${DATASETS[@]}"; do
  echo
  echo "============================================================"
  echo "[1/2] VANILLA tune | ds=${ds} enc=${ENCODER}"
  echo "============================================================"
  python tune_optuna.py \
    --model vanilla \
    --dataset "${ds}" --encoder "${ENCODER}" \
    --attack none \
    --objective auc_f1 --tune_scope gnn \
    --n_trials "${N_TRIALS_VANILLA}" \
    --start_seed "${START_SEED}" --seed_num "${TUNE_SEEDS}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --log_root "${LOG_ROOT}" \
    --tag "${TAG}_vanilla"

  BEST_V="$(latest_best "${ds}" "${ENCODER}" "vanilla" "auc_f1")"
  if [[ -z "${BEST_V}" ]]; then
    echo "[ERROR] Cannot find vanilla best_overall.json for ds=${ds} enc=${ENCODER}"
    exit 1
  fi
  echo "[INFO] vanilla best: ${BEST_V}"

  echo
  echo "============================================================"
  echo "[2/2] EDGE_ADDER tune | ds=${ds} enc=${ENCODER}"
  echo "============================================================"
  python tune_optuna.py \
    --model edge_adder \
    --dataset "${ds}" --encoder "${ENCODER}" \
    --attack none \
    --objective auc_f1_balanced --w_dp "${W_DP}" --w_eo "${W_EO}" \
    --tune_scope both \
    --n_trials "${N_TRIALS_EDGE}" \
    --start_seed "${START_SEED}" --seed_num "${TUNE_SEEDS}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" --pretrain_epochs "${PRETRAIN_EPOCHS}" --edge_epochs "${EDGE_EPOCHS}" \
    --edge_pipeline freeze_gnn_then_edge \
    --pretrain_lambda_dp 0 --pretrain_lambda_eo 0 \
    --best_overall_path "${BEST_V}" \
    --log_root "${LOG_ROOT}" \
    --tag "${TAG}_edgeadder"
done

echo
echo "✅ Done. Logs under: ${LOG_ROOT}"
