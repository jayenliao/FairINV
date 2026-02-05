#!/usr/bin/env bash
set -euo pipefail

# Repo root detection (works even if scripts live under commands/adv)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer git to locate repo top
if command -v git >/dev/null 2>&1 && git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
  # Fallback: walk upward until train.py is found
  REPO_ROOT="${SCRIPT_DIR}"
  while [[ "${REPO_ROOT}" != "/" && ! -f "${REPO_ROOT}/train.py" ]]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
  done
  if [[ ! -f "${REPO_ROOT}/train.py" ]]; then
    echo "[ERROR] Could not find repo root containing train.py. Set TRAIN_PY explicitly." >&2
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_PY="${TRAIN_PY:-${REPO_ROOT}/train.py}"

# Default log dir (keep your current convention if you want)
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/commands/logs_suite}"

# Default runtime knobs (override via env vars)
GPU_ID="${GPU_ID:-}"
DEVICE="${DEVICE:-cuda}"
NUM_THREADS="${NUM_THREADS:-4}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs_suite}"

# Default experiment knobs (override via env vars)
EPOCHS="${EPOCHS:-300}"
SEED_NUM="${SEED_NUM:-3}"
START_SEED="${START_SEED:-42}"
LAMBDA_DP="${LAMBDA_DP:-0.0}"
LAMBDA_EO="${LAMBDA_EO:-0.0}"
EO_MODE="${EO_MODE:-both}"

BEST_OVERALL_PATH="${BEST_OVERALL_PATH:-}"  # optional path to best_overall.json

# NIFA defaults (override via env vars)
NIFA_NODE="${NIFA_NODE:-102}"
NIFA_EDGE="${NIFA_EDGE:-50}"
NIFA_ALPHA="${NIFA_ALPHA:-1.0}"
NIFA_BETA="${NIFA_BETA:-1.0}"
NIFA_RATIO="${NIFA_RATIO:-0.5}"
NIFA_MODE="${NIFA_MODE:-uncertainty}"
NIFA_EPOCHS="${NIFA_EPOCHS:-1000}"
NIFA_LR="${NIFA_LR:-0.001}"
NIFA_LOOPS="${NIFA_LOOPS:-50}"
NIFA_KEEP_MARKERS="${NIFA_KEEP_MARKERS:-0}"

DRY_RUN="${DRY_RUN:-0}"

has_flag() {
  local flag="$1"
  "${PYTHON_BIN}" "${TRAIN_PY}" -h 2>/dev/null | grep -q -- "${flag}"
}

maybe_cuda_visible() {
  if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi
}

maybe_best_overall() {
  if [[ -n "${BEST_OVERALL_PATH}" ]] && has_flag "--best_overall_path"; then
    echo "--best_overall_path" "${BEST_OVERALL_PATH}"
  fi
}

maybe_advtrain_flags() {
  # If your repo includes the adv-train patch (flags like --advtrain), this function enables it.
  # Otherwise returns empty.
  local mode="${1:-mix}"       # mix | robust
  local k="${2:-2}"
  local reduce="${3:-max}"     # mean|max|logsumexp
  local tau="${4:-0.5}"
  local mix_lambda="${5:-1.0}"

  if ! has_flag "--advtrain"; then
    return 0
  fi

  if [[ "${mode}" == "mix" ]]; then
    echo "--advtrain" "--advtrain_mode" "mix" \
         "--advtrain_k" "${k}" \
         "--advtrain_mix_lambda" "${mix_lambda}"
  else
    echo "--advtrain" "--advtrain_mode" "robust" \
         "--advtrain_k" "${k}" \
         "--advtrain_reduce" "${reduce}" \
         "--advtrain_tau" "${tau}" \
         "--advtrain_include_clean"
  fi
}

maybe_nifa_gamma_flag() {
  # If your repo includes --nifa_gamma, pass it. Otherwise ignore.
  local gamma="${1:-1.0}"
  if has_flag "--nifa_gamma"; then
    echo "--nifa_gamma" "${gamma}"
  fi
}

run_train() {
  # Usage:
  #   run_train <dataset> <encoder> <model> <attack> <attack_when> [extra args...]
  local dataset="$1"; shift
  local encoder="$1"; shift
  local model="$1"; shift
  local attack="$1"; shift
  local attack_when="$1"; shift

  maybe_cuda_visible

  # Build args
  local args=(
    "--dataset" "${dataset}"
    "--encoder" "${encoder}"
    "--model" "${model}"
    "--device" "${DEVICE}"
    "--num_threads" "${NUM_THREADS}"
    "--epochs" "${EPOCHS}"
    "--seed_num" "${SEED_NUM}"
    "--start_seed" "${START_SEED}"
    "--log_dir" "${LOG_ROOT}"
    "--lambda_dp" "${LAMBDA_DP}"
    "--lambda_eo" "${LAMBDA_EO}"
    "--eo_mode" "${EO_MODE}"
    "--attack" "${attack}"
    "--attack_when" "${attack_when}"
  )

  # Optionally load optuna best_overall
  read -r -a bo <<<"$(maybe_best_overall || true)"
  args+=("${bo[@]}")

  # NIFA knobs (only if attack is nifa or your extra args rely on them)
  if [[ "${attack}" == "nifa" ]]; then
    args+=(
      "--nifa_node" "${NIFA_NODE}"
      "--nifa_edge" "${NIFA_EDGE}"
      "--nifa_alpha" "${NIFA_ALPHA}"
      "--nifa_beta" "${NIFA_BETA}"
      "--nifa_ratio" "${NIFA_RATIO}"
      "--nifa_mode" "${NIFA_MODE}"
      "--nifa_epochs" "${NIFA_EPOCHS}"
      "--nifa_lr" "${NIFA_LR}"
      "--nifa_loops" "${NIFA_LOOPS}"
    )
    if [[ "${NIFA_KEEP_MARKERS}" == "1" ]]; then
      args+=("--nifa_keep_markers")
    fi
  fi

  # Extra args
  args+=("$@")

  echo
  echo "=============================="
  echo "RUN  dataset=${dataset} encoder=${encoder} model=${model} attack=${attack} when=${attack_when}"
  echo "LOG  ${LOG_ROOT}"
  echo "CMD  ${PYTHON_BIN} ${TRAIN_PY} ${args[*]}"
  echo "=============================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "${PYTHON_BIN}" "${TRAIN_PY}" "${args[@]}"
}
