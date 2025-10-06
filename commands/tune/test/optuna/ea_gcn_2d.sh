#!/usr/bin/env bash
set -e

DATASETS=(german)
ENCODERS=(gcn)
MODELS=(edge_adder)

N_THREADS=${N_THREADS:-8}
N_TRIALS=${N_TRIALS:-4}                 # #trials per scenario to start
EPOCHS=${EPOCHS:-500}                    # epochs per trial
SEEDS="${SEEDS:-0 1 2}"                  # 3 seeds default
OBJ=${OBJ:-balanced}                     # f1|auc|balanced
BAL_ON=${BAL_ON:-auc}                    # when balanced
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}

for ds in "${DATASETS[@]}"; do
  for enc in "${ENCODERS[@]}"; do
    for m in "${MODELS[@]}"; do
      echo "=== ${m} / ${enc} / ${ds} ==="
      python tune_optuna.py \
        --model "$m" --encoder "$enc" --dataset "$ds" \
        --seeds ${SEEDS} --num_threads ${N_THREADS} --epochs ${EPOCHS} \
        --objective ${OBJ} --balanced_on ${BAL_ON} --w_dp ${W_DP} --w_eo ${W_EO} \
        --n_trials ${N_TRIALS} --sampler tpe --pruner median \
        --tag "auto75"
    done
  done
done
