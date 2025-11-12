#!/usr/bin/env bash
# gcn.sh — Optuna tuning for NIFA attacks on vanilla GCN (with --dry support)

set -Eeuo pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

# ---------- CLI flags ----------
DRY=${DRY:-0}
show_help() {
  cat <<'H'
Usage: bash gcn.sh [--dry|-n] [--help|-h]

Flags:
  --dry, -n   Print commands without executing them.
  --help, -h  Show this help.

You can also set DRY=1 in the environment.
H
}

# Parse only our flags (script uses env vars for the rest)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry|-n) DRY=1; shift;;
    --help|-h) show_help; exit 0;;
    *) echo "Unknown argument: $1"; echo; show_help; exit 2;;
  esac
done

# Helper to echo-or-run commands
run_cmd() {
  if [[ "${DRY}" == "1" ]]; then
    printf '[DRY] '
    printf '%q ' "$@"
    echo
    return 0
  else
    "$@"
  fi
}

# ---------- Defaults (override via env) ----------
DATASETS=(${DATASETS:-bail pokec_n pokec_z nba german})
MODELS=(${MODELS:-vanilla})
ENCODERS=(${ENCODERS:-gcn})

START_SEED=${START_SEED:-0}
SEED_NUM=${SEED_NUM:-10}
N_TRIALS=${N_TRIALS:-64}
EPOCHS=${EPOCHS:-500}
N_THREADS=${N_THREADS:-4}

OBJ=${OBJ:-attack_balanced}
BAL_ON=${BAL_ON:-f1}
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}

UTIL_ON=${UTIL_ON:-f1}
UTIL_MIN=${UTIL_MIN:-0.55}
LAMBDA_UTIL=${LAMBDA_UTIL:-1.0}

LOG_ROOT=${LOG_ROOT:-logs/tune_vanilla_nifa}
SAMPLER=${SAMPLER:-tpe}
PRUNER=${PRUNER:-median}
STORAGE=${STORAGE:-}
TAG=${TAG:-nifa}

declare -A best_paths=(
  [bail]="best_overall_json/vanilla/gcn/bail.json"
  [pokec_z]="best_overall_json/vanilla/gcn/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/gcn/pokec_n.json"
  [nba]="best_overall_json/vanilla/gcn/nba.json"
  [german]="best_overall_json/vanilla/gcn/german.json"
)

echo "== NIFA tuning with Optuna =="
echo "GPU(s):            ${CUDA_VISIBLE_DEVICES}"
echo "Datasets:          ${DATASETS[*]}"
echo "Model/Encoder:     ${MODELS[*]} / ${ENCODERS[*]}"
# echo "Seeds:             ${SEEDS}"
echo "Trials per job:    ${N_TRIALS}"
echo "Epochs per trial:  ${EPOCHS}"
echo "Threads:           ${N_THREADS}"
echo "Objective:         ${OBJ} (balanced_on=${BAL_ON}, w_dp=${W_DP}, w_eo=${W_EO})"
[[ -n "${UTIL_MIN:-}" ]] && echo "Utility guardrail: ${UTIL_ON} >= ${UTIL_MIN} (lambda=${LAMBDA_UTIL})"
echo "Log root:          ${LOG_ROOT}"
echo "Sampler/Pruner:    ${SAMPLER} / ${PRUNER}"
[[ -n "${STORAGE}" ]] && echo "Optuna storage:    ${STORAGE}"
echo "Tag:               ${TAG}"
echo "Dry run:           ${DRY}"

for ds in "${DATASETS[@]}"; do
    echo "============ ${ds^^} ============="
    best_path="${best_paths[$ds]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $ds at $best_path"
    else
        echo "Using best_overall.json from: $best_path"
    fi
    echo
done

for ds in "${DATASETS[@]}"; do
  best_path="${best_paths[$ds]}"
  for enc in "${ENCODERS[@]}"; do
    for m in "${MODELS[@]}"; do
      echo "---- ${m} / ${enc} / ${ds} ----"
      cmd=(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python tune_optuna.py
        --model "${m}" --encoder "${enc}" --dataset "${ds}"
        --best_overall_path "${best_path}"
        --attack nifa --tune_scope attack
        --start_seed ${START_SEED} --seed_num ${SEED_NUM}
        --num_threads ${N_THREADS}
        --epochs ${EPOCHS}
        --objective "${OBJ}" --balanced_on "${BAL_ON}" --w_dp "${W_DP}" --w_eo "${W_EO}"
        --util_on "${UTIL_ON}" --lambda_util "${LAMBDA_UTIL}"
        --n_trials ${N_TRIALS}
        --sampler "${SAMPLER}" --pruner "${PRUNER}"
        --log_root "${LOG_ROOT}"
        --tag "${TAG}"
      )
      [[ -n "${STORAGE}"   ]] && cmd+=( --storage "${STORAGE}" )
      [[ -n "${UTIL_MIN:-}" ]] && cmd+=( --util_min "${UTIL_MIN}" )

      echo "${cmd[*]}"
      run_cmd "${cmd[@]}"
      echo
    done
  done
done

echo "✅ ${DRY:+(dry) }All NIFA tuning jobs finished."
