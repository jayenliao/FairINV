set -euo pipefail
CUDA_VISIBLE_DEVICES=2

echo "Attacking Vanilla sgc for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_vanilla_paths=(
  [bail]="best_overall_json/vanilla/sgc/bail.json"
  [pokec_z]="best_overall_json/vanilla/sgc/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/sgc/pokec_n.json"
  [nba]="best_overall_json/vanilla/sgc/nba.json"
  [german]="best_overall_json/vanilla/sgc/german.json"
)

declare -A best_nifa_paths=(
  [bail]="best_overall_json/vanilla_nifa/sgc/bail.json"
  [pokec_z]="best_overall_json/vanilla_nifa/sgc/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla_nifa/sgc/pokec_n.json"
  [nba]="best_overall_json/vanilla_nifa/sgc/nba.json"
  [german]="best_overall_json/vanilla_nifa/sgc/german.json"
)

declare -A best_ea_paths=(
  [bail]="best_overall_json/ea/sgc/bail.json"
  [pokec_z]="best_overall_json/ea/sgc/pokec_z.json"
  [pokec_n]="best_overall_json/ea/sgc/pokec_n.json"
  [nba]="best_overall_json/ea/sgc/nba.json"
  [german]="best_overall_json/ea/sgc/german.json"
)

# Common args
encoder="sgc"
model="edge_adder"
attack="nifa"
epochs=1000
start_seed=0
seed_num=10
log_dir="logs/use_tuned_vanilla_hp_run_ea/tuned_ea/tuned_nifa/"

# Loop over datasets

for dataset in bail pokec_z pokec_n german nba; do
    echo
    echo "============ ${dataset^^} ============="
    best_vanilla_path="${best_vanilla_paths[$dataset]}"
    best_nifa_path="${best_nifa_paths[$dataset]}"
    best_ea_path="${best_ea_paths[$dataset]}"
    if [[ ! -f "$best_vanilla_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_vanilla_path"
    fi
    if [[ ! -f "$best_nifa_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_nifa_path"
    fi
    if [[ ! -f "$best_ea_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_ea_path"
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model "$model" \
        --encoder "$encoder" \
        --dataset "$dataset" \
        --start_seed "$start_seed" \
        --seed_num "$seed_num" \
        --epochs "$epochs" \
        --best_overall_path "$best_vanilla_path" "$best_nifa_path" "$best_ea_path" \
        --log_dir "$log_dir" \
        --attack "$attack"
done

echo
echo "✅ All runs finished."
