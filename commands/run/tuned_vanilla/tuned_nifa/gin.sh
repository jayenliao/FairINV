set -euo pipefail
CUDA_VISIBLE_DEVICES=3

echo "Attacking Vanilla GIN for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_vanilla_paths=(
  [bail]="best_overall_json/vanilla/gin/bail.json"
  [pokec_z]="best_overall_json/vanilla/gin/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/gin/pokec_n.json"
  [nba]="best_overall_json/vanilla/gin/nba.json"
  [german]="best_overall_json/vanilla/gin/german.json"
)

declare -A best_nifa_paths=(
  [bail]="best_overall_json/vanilla_nifa/gin/bail.json"
  [pokec_z]="best_overall_json/vanilla_nifa/gin/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla_nifa/gin/pokec_n.json"
  [nba]="best_overall_json/vanilla_nifa/gin/nba.json"
  [german]="best_overall_json/vanilla_nifa/gin/german.json"
)

# Common args
encoder="gin"
model="vanilla"
attack="nifa"
epochs=1000
start_seed=0
seed_num=10
lambda_dp=0.0
lambda_eo=0.0
log_dir="logs/tuned_vanilla/tuned_nifa/"

# Loop over datasets

for dataset in bail pokec_z pokec_n german nba; do
    echo
    echo "============ ${dataset^^} ============="
    best_vanilla_path="${best_vanilla_paths[$dataset]}"
    best_nifa_path="${best_nifa_paths[$dataset]}"
    best_path=("$best_vanilla_path" "$best_nifa_path")
    if [[ ! -f "$best_vanilla_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_vanilla_path"
    fi
    if [[ ! -f "$best_nifa_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_nifa_path"
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model "$model" \
        --encoder "$encoder" \
        --dataset "$dataset" \
        --start_seed "$start_seed" \
        --seed_num "$seed_num" \
        --epochs "$epochs" \
        --best_overall_path "$best_vanilla_path" "$best_nifa_path" \
        --lambda_dp "$lambda_dp" \
        --lambda_eo "$lambda_eo" \
        --log_dir "$log_dir" \
        --attack "$attack"
done

echo
echo "✅ All runs finished."
