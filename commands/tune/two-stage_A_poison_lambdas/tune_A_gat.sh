#!/usr/bin/env bash
set -euo pipefail

# Minimal Optuna tuning for pipeline-A (freeze GNN, then train EdgeAdder)
# under NIFA (degree) poisoning.
#
# What it does:
# - Loads your tuned baseline HPs via --best_overall_path (typically: vanilla + vanilla_nifa).
# - Tunes ONLY a small subset of HPs that are most likely to change when switching to pipeline-A:
#     lambda_dp, lambda_eo, lambda_edge_l1, edge_k
#   (Optionally you can also include: pretrain_lambda_dp, pretrain_lambda_eo)
#
# How to run:
#   CUDA_VISIBLE_DEVICES=0 bash tune_A_poison_lambdas.sh

# -------------------- config --------------------
DATASETS=(pokec_z pokec_n bail nba german)
ENCODERS=(gat)

# Tuning budget
N_TRIALS=64
TUNE_EPOCHS=500   # use fewer epochs for tuning; rerun final with --epochs 1000
START_SEED=0
SEED_NUM=10        # 10 seeds for tuning stability; rerun final with 10

# Logging
LOG_ROOT="logs/optuna_A"
TAG="A_freeze_poison_lambdas"

# Attack settings (tuned NIFA params can still be loaded from best_overall_path)
ATTACK="nifa"
ATTACK_WHEN="train"   # poisoning
NIFA_MODE="degree"

# Pipeline-A: stage-1 clean objective + stage-3 edge L1 regularizer
EDGE_PIPELINE="freeze_gnn_then_edge"
PRETRAIN_LAMBDA_DP=0
PRETRAIN_LAMBDA_EO=0

# Objective used to pick the best trial
OBJECTIVE="auc_f1_balanced"
BALANCED_ON="f1"
W_DP=1.0
W_EO=1.0

# Tune only these keys
TUNE_SUBSET="lambda_dp,lambda_eo,lambda_edge_l1,edge_k"

# Baseline tuned JSON layout (edit if your paths differ)
BASE_VANILLA_DIR="best_overall_json/vanilla"
# BASE_NIFA_DIR="best_overall_json/vanilla_nifa"

# -------------------- run --------------------
for encoder in "${ENCODERS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    base_vanilla_path="${BASE_VANILLA_DIR}/${encoder}/${dataset}.json"
    # base_nifa_path="${BASE_NIFA_DIR}/${encoder}/${dataset}.json"

    echo
    echo "============ Tuning ${dataset^^} / ${encoder^^} ============"

    if [[ ! -f "${base_vanilla_path}" ]]; then
      echo "⚠️  Skip: missing ${base_vanilla_path}"
      continue
    fi
    # if [[ ! -f "${base_nifa_path}" ]]; then
    #   echo "⚠️  Skip: missing ${base_nifa_path}"
    #   continue
    # fi

    python tune_optuna.py \
      --model edge_adder \
      --encoder "${encoder}" \
      --dataset "${dataset}" \
      --attack "${ATTACK}" \
      --attack_when "${ATTACK_WHEN}" \
      --nifa_mode "${NIFA_MODE}" \
      --edge_pipeline "${EDGE_PIPELINE}" \
      --pretrain_lambda_dp "${PRETRAIN_LAMBDA_DP}" \
      --pretrain_lambda_eo "${PRETRAIN_LAMBDA_EO}" \
      --objective "${OBJECTIVE}" \
      --balanced_on "${BALANCED_ON}" \
      --w_dp "${W_DP}" \
      --w_eo "${W_EO}" \
      --epochs "${TUNE_EPOCHS}" \
      --start_seed "${START_SEED}" \
      --seed_num "${SEED_NUM}" \
      --tune_scope both \
      --tune_subset "${TUNE_SUBSET}" \
      --best_overall_path "${base_vanilla_path}" \
      --n_trials "${N_TRIALS}" \
      --log_root "${LOG_ROOT}" \
      --tag "${TAG}"
  done
done

echo
echo "✅ All tuning runs finished."
