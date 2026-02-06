#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Robust repo root detection ---
if command -v git >/dev/null 2>&1 && git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
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

# Default runtime knobs (override via env vars)
GPU_ID="${GPU_ID:-}"
DEVICE="${DEVICE:-cuda}"
NUM_THREADS="${NUM_THREADS:-4}"

# Default log dir:
# - if <repo>/commands exists => <repo>/commands/logs_suite
# - else => <repo>/logs_suite
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ -d "${REPO_ROOT}/commands" ]]; then
    LOG_ROOT="${REPO_ROOT}/commands/logs_suite"
  else
    LOG_ROOT="${REPO_ROOT}/logs_suite"
  fi
fi

# Default experiment knobs (override via env vars)
EPOCHS="${EPOCHS:-300}"
SEED_NUM="${SEED_NUM:-3}"
START_SEED="${START_SEED:-42}"
LAMBDA_DP="${LAMBDA_DP:-0.0}"
LAMBDA_EO="${LAMBDA_EO:-0.0}"
EO_MODE="${EO_MODE:-both}"

# best_overall JSONs:
# - BEST_OVERALL_PATHS: space-separated list (e.g., "victim.json nifa.json")
# - BEST_OVERALL_PATH : single file (legacy)
BEST_OVERALL_PATHS="${BEST_OVERALL_PATHS:-}"
BEST_OVERALL_PATH="${BEST_OVERALL_PATH:-}"
BEST_OVERALL_MERGE_ALL="${BEST_OVERALL_MERGE_ALL:-0}"  # if your args support it

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
NIFA_GAMMA="${NIFA_GAMMA:-1.0}"   # only passed if --nifa_gamma exists

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

  # --- best_overall JSON(s) ---
  # Prefer BEST_OVERALL_PATHS (multi) over BEST_OVERALL_PATH (single)
  if has_flag "--best_overall_path"; then
    if [[ -n "${BEST_OVERALL_PATHS}" ]]; then
      # shellcheck disable=SC2206
      local bo_arr=(${BEST_OVERALL_PATHS})
      args+=("--best_overall_path" "${bo_arr[@]}")
      if [[ "${BEST_OVERALL_MERGE_ALL}" == "1" ]] && has_flag "--best_overall_merge_all"; then
        args+=("--best_overall_merge_all")
      fi
    elif [[ -n "${BEST_OVERALL_PATH}" ]]; then
      args+=("--best_overall_path" "${BEST_OVERALL_PATH}")
      if [[ "${BEST_OVERALL_MERGE_ALL}" == "1" ]] && has_flag "--best_overall_merge_all"; then
        args+=("--best_overall_merge_all")
      fi
    fi
  fi

  # NIFA knobs (only if attack is nifa)
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

    # Optional utility weight if supported
    read -r -a gg <<<"$(maybe_nifa_gamma_flag "${NIFA_GAMMA}" || true)"
    args+=("${gg[@]}")
  fi

  # Extra args
  args+=("$@")

  echo
  echo "=============================="
  echo "REPO ${REPO_ROOT}"
  echo "RUN  dataset=${dataset} encoder=${encoder} model=${model} attack=${attack} when=${attack_when}"
  echo "LOG  ${LOG_ROOT}"
  echo "CMD  ${PYTHON_BIN} ${TRAIN_PY} ${args[*]}"
  echo "=============================="

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  "${PYTHON_BIN}" "${TRAIN_PY}" "${args[@]}"
}
