#!/usr/bin/env bash
set -e

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

# ============= Vanilla backbones =============

python tune.py --model vanilla --encoder gcn --dataset german \
  --seeds $SEEDS --epochs 500 \
  --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
  --objective balanced --balanced_on f1 --w_dp 1.0 --w_eo 1.0 --tag vanilla_gcn_balanced

# python tune.py --model vanilla --encoder gcn --dataset bail \
#   --seeds $SEEDS --epochs 500 \
#   --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
#   --objective balanced --balanced_on f1 --w_dp 1.0 --w_eo 1.0 --tag vanilla_gcn_balanced

# python tune.py --model vanilla --encoder gcn --dataset pokec_z \
#   --seeds $SEEDS --epochs 500 \
#   --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
#   --objective balanced --balanced_on f1 --w_dp 1.0 --w_eo 1.0 --tag vanilla_gcn_balanced

# python tune.py --model vanilla --encoder gcn --dataset pokec_n \
#   --seeds $SEEDS --epochs 500 \
#   --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
#   --objective balanced --balanced_on f1 --w_dp 1.0 --w_eo 1.0 --tag vanilla_gcn_balanced

# python tune.py --model vanilla --encoder gcn --dataset nba \
#   --seeds $SEEDS --epochs 500 \
#   --lr_list 1e-2 5e-3 1e-3 --hid_dim_list 16 32 64 --weight_decay_list 1e-5 5e-5 \
#   --objective balanced --balanced_on f1 --w_dp 1.0 --w_eo 1.0 --tag vanilla_gcn_balanced
