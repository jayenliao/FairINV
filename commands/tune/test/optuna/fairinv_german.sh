#!/usr/bin/env bash
# run_optuna_all.sh — iterate the 75 scenarios (3 models × 5 backbones × 5 datasets)
# Adjust N_TRIALS/SEEDS to fit your budget.

set -e

DATASETS=(german)
ENCODERS=(gcn gat sage sgc gin)
MODELS=(fairinv)

N_THREADS=${N_THREADS:-16}
N_TRIALS=${N_TRIALS:-4}                  # #trials per scenario to start
EPOCHS=${EPOCHS:-500}                    # epochs per trial
SEEDS="${SEEDS:-0 1 2}"                  # 3 seeds default
OBJ=${OBJ:-f1}                           # f1|auc|balanced
BAL_ON=${BAL_ON:-f1}                     # when balanced
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}

for ds in "${DATASETS[@]}"; do
  for enc in "${ENCODERS[@]}"; do
    for m in "${MODELS[@]}"; do
      echo "=== ${m} / ${enc} / ${ds} ==="
      python tune_optuna.py \
        --model "$m" --encoder "$enc" --dataset "$ds" \
        --num_threads ${N_THREADS} --seeds ${SEEDS} --epochs ${EPOCHS} \
        --objective ${OBJ} --balanced_on ${BAL_ON} --w_dp ${W_DP} --w_eo ${W_EO} \
        --n_trials ${N_TRIALS} --sampler tpe --pruner none \
        --tag "auto75"
    done
  done
done
