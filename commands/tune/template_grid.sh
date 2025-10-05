#!/usr/bin/env bash
# run_tune_template.sh — examples of how to use tune.py
set -e

DATASET=${DATASET:-german}
SEEDS=${SEEDS:-"42 43 44 45 46"}

# ============= Vanilla backbones =============
python tune.py --model vanilla --encoder gcn --dataset "$DATASET" \
  --seeds $SEEDS --epochs 500 \
  --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
  --objective balanced --balanced_on auc --w_dp 1.0 --w_eo 1.0 --tag vanilla_gcn_balanced

python tune.py --model vanilla --encoder gin --dataset "$DATASET" \
  --seeds $SEEDS --epochs 500 \
  --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
  --objective f1 --tag vanilla_gin_f1

# ============= FairINV (various backbones) =============
python tune.py --model fairinv --encoder gat --dataset "$DATASET" \
  --seeds $SEEDS --epochs 800 \
  --alpha_list 1 5 10 --lr_sp_list 0.05 0.1 0.2 --env_num_list 2 3 \
  --lr_list 1e-2 5e-3 --hid_dim_list 16 32 \
  --objective balanced --balanced_on auc --w_dp 1.0 --w_eo 1.0 --tag fairinv_gat_balanced

# ============= EdgeAdder (fairness weights & k) =============
python tune.py --model edge_adder --encoder gcn --dataset "$DATASET" \
  --seeds $SEEDS --epochs 600 \
  --edge_k_list 1 2 3 --lambda_dp_list 0.05 0.1 0.2 --lambda_edge_l1_list 1e-4 5e-4 1e-3 \
  --lr_list 1e-2 5e-3 \
  --objective balanced --balanced_on auc --w_dp 1.0 --w_eo 1.0 --tag edgeadder_gcn_balanced

echo "Done. See logs/tune/"
