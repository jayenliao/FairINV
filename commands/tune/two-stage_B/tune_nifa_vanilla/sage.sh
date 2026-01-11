#!/usr/bin/env bash
set -euo pipefail

# =========================
# Experiment B: NIFA eval-only (attack applied ONLY at evaluation)
# Optuna tunes NIFA hyperparameters only (tune_scope=attack).
# Requires patch: train.py logs val_attack, tune_optuna.py supports --study_on val_attack.
# =========================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# ---- Victim model/backbone ----
MODEL=${MODEL:-vanilla}     # vanilla | fairinv | edge_adder | edge_minmax
ENCODER=${ENCODER:-sage}        # gcn | gat | gin | sage | sgc

# ---- Optuna settings ----
N_TRIALS=${N_TRIALS:-32}
SEED_NUM=${SEED_NUM:-10}
START_SEED=${START_SEED:-0}
EPOCHS=${EPOCHS:-500}

# ---- Attack objective (maximize attack strength) ----
OBJECTIVE=${OBJECTIVE:-attack_balanced}
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}
UTIL_ON=${UTIL_ON:-f1}
UTIL_MIN=${UTIL_MIN:-0.60}
LAMBDA_UTIL=${LAMBDA_UTIL:-10.0}

# ---- Logs ----
LOG_ROOT=${LOG_ROOT:-logs/optuna_nifa_expB}
TAG=${TAG:-expB_nifa_eval}

# ---- Datasets ----
datasets=(${DATASETS:-pokec_z pokec_n german bail nba})

# (Optional) Fix victim/defense hyperparams from previous best_overall.json.
declare -A best_paths=(
    ["german"]="best_overall_json/optuna_big/vanilla/sage/german.json"
    ["bail"]="best_overall_json/optuna_big/vanilla/sage/bail.json"
    ["nba"]="best_overall_json/optuna_big/vanilla/sage/nba.json"
    ["pokec_z"]="best_overall_json/optuna_big/vanilla/sage/pokec_z.json"
    ["pokec_n"]="best_overall_json/optuna_big/vanilla/sage/pokec_n.json"
)

for ds in "${datasets[@]}"; do
  echo
  echo "=============================="
  echo "Exp-B NIFA Tune | dataset=${ds} | model=${MODEL} | encoder=${ENCODER}"
  echo "=============================="

  extra_best=()
  if [[ -n "${best_paths[$ds]:-}" ]]; then
    if [[ -f "${best_paths[$ds]}" ]]; then
      extra_best+=(--best_overall_path "${best_paths[$ds]}")
      echo "[info] Using best_overall_path=${best_paths[$ds]}"
    else
      echo "[warn] best_overall.json not found at: ${best_paths[$ds]} (ignored)"
    fi
  fi

  python tune_optuna.py \
    --dataset "${ds}" \
    --model "${MODEL}" \
    --encoder "${ENCODER}" \
    --epochs "${EPOCHS}" \
    --num_threads "${OMP_NUM_THREADS}" \
    --attack nifa \
    --attack_when eval \
    --tune_scope attack \
    --objective "${OBJECTIVE}" \
    --w_dp "${W_DP}" --w_eo "${W_EO}" \
    --util_on "${UTIL_ON}" --util_min "${UTIL_MIN}" --lambda_util "${LAMBDA_UTIL}" \
    --seed_num "${SEED_NUM}" --start_seed "${START_SEED}" \
    --n_trials "${N_TRIALS}" \
    --study_on val_attack \
    --log_root "${LOG_ROOT}" \
    --tag "${TAG}" \
    "${extra_best[@]}"
done
