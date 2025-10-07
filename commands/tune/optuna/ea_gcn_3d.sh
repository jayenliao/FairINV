#!/usr/bin/env bash
set -e

DATASETS=(bail pokec_z pokec_n)
ENCODERS=(gcn)
MODELS=(edge_adder)

N_THREADS=${N_THREADS:-5}
N_TRIALS=${N_TRIALS:-64}                 # trials per scenario to start
EPOCHS=${EPOCHS:-1000}                  # epochs per trial
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
OBJ=${OBJ:-auc_f1_mean_minus_std}
BAL_ON=${BAL_ON:-f1}                    # when balanced
W_DP=${W_DP:-1.0}
W_EO=${W_EO:-1.0}

for ds in "${DATASETS[@]}"; do
  for enc in "${ENCODERS[@]}"; do
    for m in "${MODELS[@]}"; do
      echo "=== ${m} / ${enc} / ${ds} ==="
      python tune_optuna_robust.py \
        --model "$m" --encoder "$enc" --dataset "$ds" \
        --seeds ${SEEDS} --num_threads ${N_THREADS} --epochs ${EPOCHS} \
        --objective ${OBJ} --balanced_on ${BAL_ON} --w_dp ${W_DP} --w_eo ${W_EO} \
        --n_trials ${N_TRIALS} --sampler tpe --pruner median \
        --log_root "logs/optuna" --tag "robust"
    done
  done
done
