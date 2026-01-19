#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# Choose the victim used for tuning the attack HPs.
# If you want "attacker targets vanilla", keep this as vanilla.
NIFA_VICTIM_MODEL=${NIFA_VICTIM_MODEL:-vanilla}   # vanilla | edge_adder | fairinv | edge_minmax
ENCODER=${ENCODER:-gcn}

N_TRIALS=${N_TRIALS:-32}
SEED_NUM=${SEED_NUM:-5}
START_SEED=${START_SEED:-42}

OBJECTIVE=${OBJECTIVE:-attack_balanced}
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}
UTIL_ON=${UTIL_ON:-f1}
UTIL_MIN=${UTIL_MIN:-0.60}
LAMBDA_UTIL=${LAMBDA_UTIL:-10.0}

datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

# Where optuna writes studies (machine-local)
LOG_ROOT=${LOG_ROOT:-logs/optuna_nifa_expA_train}
TAG=${TAG:-expA_train_tune_attack}

# Where to export stable JSONs (portable)
EXPORT_DIR=${EXPORT_DIR:-best_overall_json/optuna_nifa_expA/vanilla_expA/${ENCODER}}
mkdir -p "${EXPORT_DIR}"

# (Optional but recommended) fix victim hyperparams during NIFA tuning
# by loading victim's clean best_overall (so only NIFA changes).
declare -A victim_best_paths=(
  ["german"]="best_overall_json/optuna_big/${NIFA_VICTIM_MODEL}/${ENCODER}/german.json"
  ["bail"]="best_overall_json/optuna_big/${NIFA_VICTIM_MODEL}/${ENCODER}/bail.json"
  ["nba"]="best_overall_json/optuna_big/${NIFA_VICTIM_MODEL}/${ENCODER}/nba.json"
  ["pokec_z"]="best_overall_json/optuna_big/${NIFA_VICTIM_MODEL}/${ENCODER}/pokec_z.json"
  ["pokec_n"]="best_overall_json/optuna_big/${NIFA_VICTIM_MODEL}/${ENCODER}/pokec_n.json"
)

for ds in "${datasets[@]}"; do
  echo
  echo "=============================="
  echo "Tune NIFA (ExpA train-poison) | victim=${NIFA_VICTIM_MODEL}/${ENCODER} | ds=${ds}"
  echo "=============================="

  vb="${victim_best_paths[$ds]}"
  extra_best=()
  if [[ -f "${vb}" ]]; then
    extra_best+=(--best_overall_path "${vb}")
    echo "[info] Fixing victim HPs using: ${vb}"
  else
    echo "[warn] victim best_overall not found (tuning will use defaults): ${vb}"
  fi

  python tune_optuna.py \
    --dataset "${ds}" \
    --model "${NIFA_VICTIM_MODEL}" \
    --encoder "${ENCODER}" \
    --attack nifa \
    --attack_when train \
    --tune_scope attack \
    --objective "${OBJECTIVE}" \
    --w_dp "${W_DP}" --w_eo "${W_EO}" \
    --util_on "${UTIL_ON}" --util_min "${UTIL_MIN}" --lambda_util "${LAMBDA_UTIL}" \
    --seed_num "${SEED_NUM}" --start_seed "${START_SEED}" \
    --n_trials "${N_TRIALS}" \
    --study_on val \
    --log_root "${LOG_ROOT}" \
    --tag "${TAG}" \
    "${extra_best[@]}"

  # Copy the newest best_overall.json to a stable location
  base="${LOG_ROOT}/${ds}/${ENCODER}/${NIFA_VICTIM_MODEL}/${OBJECTIVE}"
  newest="$(ls -1dt "${base}"/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${newest}" || ! -f "${newest}/best_overall.json" ]]; then
    echo "[ERROR] Could not find best_overall.json under: ${base}"
    exit 1
  fi

  cp -f "${newest}/best_overall.json" "${EXPORT_DIR}/${ds}.json"
  echo "[info] Exported: ${EXPORT_DIR}/${ds}.json"
done
