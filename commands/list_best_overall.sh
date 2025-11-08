#!/usr/bin/env bash
# list_best_overall.sh
# Usage: ./list_best_overall.sh [BASE_DIR]
# Default BASE_DIR: logs/optuna

set -euo pipefail

BASE_DIR="${1:-logs/optuna}"

datasets=(bail pokec_z pokec_n nba german)
backbones=(gcn gat gin sage sgc)
metrics=(auc_f1_mean_minus_std)
model="fairinv"

shopt -s nullglob
for bb in "${backbones[@]}"; do
  echo "===== Backbone: ${bb^^} ====="
  for ds in "${datasets[@]}"; do
    for metric in "${metrics[@]}"; do
      matches=("$BASE_DIR/$ds/$bb/$model/$metric/"*/best_overall.json)
      for f in "${matches[@]}"; do
        printf '%s\n' "$f"
      done
    done
  done
  echo
done
shopt -u nullglob
