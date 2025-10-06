#!/usr/bin/env bash
set -e

DATASETS=(german nba)
ENCODERS=(gat)
MODELS=(edge_adder)

N_THREADS=${N_THREADS:-2}
N_TRIALS=${N_TRIALS:-64}                 # trials per scenario to start
EPOCHS=${EPOCHS:-1000}                   # epochs per trial
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"  # 10 seeds default
OBJ=${OBJ:-auc_f1}                     # f1|auc|balanced
BAL_ON=${BAL_ON:-f1}                    # when balanced
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
