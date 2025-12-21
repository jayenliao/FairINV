#!/bin/bash
# sage.sh
set -euo pipefail

CUDA_VISIBLE_DEVICES=1

echo "Running vanilla sage with DP and EO loss terms for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="best_overall_json/vanilla_dp_eo/sage/bail.json"
  [pokec_z]="best_overall_json/vanilla_dp_eo/sage/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla_dp_eo/sage/pokec_n.json"
  [nba]="best_overall_json/vanilla_dp_eo/sage/nba.json"
  [german]="best_overall_json/vanilla_dp_eo/sage/german.json"
)

# Common args
encoder="sage"
model="vanilla"
epochs=1000
start_seed=0
seed_num=10
log_dir="logs/tuned_vanilla_dp_eo/no_attack"

for dataset in bail pokec_z pokec_n nba german; do
    echo
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model "$model" \
        --encoder "$encoder" \
        --dataset "$dataset" \
        --start_seed "$start_seed" \
        --seed_num "$seed_num" \
        --epochs "$epochs" \
        --best_overall_path "$best_path" \
        --log_dir "$log_dir" \
        --attack none
done


echo
echo "✅ All runs finished."
