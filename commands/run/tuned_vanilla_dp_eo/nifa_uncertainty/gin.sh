#!/bin/bash
# gin.sh
set -euo pipefail
CUDA_VISIBLE_DEVICES=0

echo "Attacking Vanilla gin for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="best_overall_json/vanilla_dp_eo/gin/bail.json"
  [pokec_z]="best_overall_json/vanilla_dp_eo/gin/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla_dp_eo/gin/pokec_n.json"
  [nba]="best_overall_json/vanilla_dp_eo/gin/nba.json"
  [german]="best_overall_json/vanilla_dp_eo/gin/german.json"
)

# Common args
encoder="gin"
model="vanilla"
epochs=1000
start_seed=0
seed_num=10
attack="nifa"
nifa_mode='uncertainty'
log_dir="logs/tuned_vanilla_dp_eo/nifa_uncertainty"

# Loop over datasets

for dataset in pokec_z pokec_n bail nba german; do
    echo
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
    fi

    if [[ "$dataset" == "pokec_n" ]]; then
        nifa_node=87
        nifa_edge=50
    elif [[ "$dataset" == "pokec_z" ]]; then
        nifa_node=102
        nifa_edge=50
    elif [[ "$dataset" == "bail" ]]; then
        nifa_node=25
        nifa_edge=50
    elif [[ "$dataset" == "nba" ]]; then
        nifa_node=4
        nifa_edge=15
    elif [[ "$dataset" == "german" ]]; then
        nifa_node=10
        nifa_edge=50
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
        --attack "$attack" \
        --nifa_mode "$nifa_mode" --nifa_node "$nifa_node" --nifa_edge "$nifa_edge" \
        --nifa_alpha 0.01 --nifa_beta 4 --nifa_ratio 0.5
done

echo
echo "✅ All runs finished."
