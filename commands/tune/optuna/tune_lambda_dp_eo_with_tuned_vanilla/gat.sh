#!/usr/bin/env bash
# gat.sh — Optuna tuning for EdgeAdder with gat backbone (using fixed tuned vanilla gat HPs)
# (with --dry support)

set -Eeuo pipefail
export CUDA_VISIBLE_DEVICES="0"

# ---------- CLI flags ----------
DRY=${DRY:-0}
show_help() {
  cat <<'H'
Usage: bash gat.sh [--dry|-n] [--help|-h]

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
DATASETS=(${DATASETS:-pokec_n nba german})
MODELS=(${MODELS:-vanilla})
ENCODERS=(${ENCODERS:-gat})
ATTACK=${ATTACK:-none}
TUNE_SCOPE=${TUNE_SCOPE:-gnn}

START_SEED=${START_SEED:-0}
SEED_NUM=${SEED_NUM:-10}
N_TRIALS=${N_TRIALS:-64}
EPOCHS=${EPOCHS:-1000}
N_THREADS=${N_THREADS:-4}

OBJ=${OBJ:-auc_f1_mean_minus_std}
BAL_ON=${BAL_ON:-f1}
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}

UTIL_ON=${UTIL_ON:-f1}
UTIL_MIN=${UTIL_MIN:-0.55}
LAMBDA_UTIL=${LAMBDA_UTIL:-1.0}

LOG_ROOT=${LOG_ROOT:-logs/tune_vanilla_dp_eo/no_attack}
SAMPLER=${SAMPLER:-tpe}
PRUNER=${PRUNER:-median}
STORAGE=${STORAGE:-}
TAG=${TAG:-vanilla-dp-eo-no-attack}
declare -A best_paths=(
  [bail]="best_overall_json/vanilla/gat/bail.json"
  [pokec_z]="best_overall_json/vanilla/gat/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/gat/pokec_n.json"
  [nba]="best_overall_json/vanilla/gat/nba.json"
  [german]="best_overall_json/vanilla/gat/german.json"
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
      cmd=( python tune_optuna.py
        --model "${m}" --encoder "${enc}" --dataset "${ds}"
        --best_overall_path "${best_path}"
        --attack ${ATTACK} --tune_scope ${TUNE_SCOPE}
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
