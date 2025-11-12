set -euo pipefail
CUDA_VISIBLE_DEVICES=1

echo "Attacking Vanilla gcn for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_vanilla_paths=(
  [bail]="best_overall_json/vanilla/gcn/bail.json"
  [pokec_z]="best_overall_json/vanilla/gcn/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/gcn/pokec_n.json"
  [nba]="best_overall_json/vanilla/gcn/nba.json"
  [german]="best_overall_json/vanilla/gcn/german.json"
)

declare -A best_nifa_paths=(
  [bail]="logs/tune_vanilla_nifa/bail/gcn/vanilla/attack_balanced/20251109-162033_nifa/best_overall.json"
  [pokec_z]="logs/tune_vanilla_nifa/pokec_z/gcn/vanilla/attack_balanced/20251110-151328_nifa/best_overall.json"
  [pokec_n]="logs/tune_vanilla_nifa/pokec_n/gcn/vanilla/attack_balanced/20251110-000134_nifa/best_overall.json"
  [nba]="logs/tune_vanilla_nifa/nba/gcn/vanilla/attack_balanced/20251111-000302_nifa/best_overall.json"
  [german]="logs/tune_vanilla_nifa/german/gcn/vanilla/attack_balanced/20251111-042921_nifa/best_overall.json"
)

# Common args
encoder="gcn"
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
