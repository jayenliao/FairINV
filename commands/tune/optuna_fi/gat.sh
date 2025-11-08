#!/usr/bin/env bash
set -e

DATASETS=(pokec_z)
ENCODERS=(gat)
MODELS=(fairinv)

N_THREADS=${N_THREADS:-16}
N_TRIALS=${N_TRIALS:-16}                 # #trials per scenario to start
EPOCHS=${EPOCHS:-500}                    # epochs per trial
SEEDS="${SEEDS:-5 6 7 8 9}"
OBJ=${OBJ:-auc_f1_mean_minus_std}
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
        --n_trials ${N_TRIALS} --sampler tpe --pruner median \
        --log_root "logs/optuna_fi" --tag "robust"
    done
  done
done
