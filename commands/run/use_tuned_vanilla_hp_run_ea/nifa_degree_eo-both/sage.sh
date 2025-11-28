set -euo pipefail
CUDA_VISIBLE_DEVICES=0

echo "Attacking EdgeAdder with sage backbone for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="best_overall_json/vanilla/sage/bail.json"
  [pokec_z]="best_overall_json/vanilla/sage/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/sage/pokec_n.json"
  [nba]="best_overall_json/vanilla/sage/nba.json"
  [german]="best_overall_json/vanilla/sage/german.json"
)

# Common args
encoder="sage"
model="edge_adder"
attack="nifa"
nifa_mode='degree'
eo_mode='both'
epochs=1000
start_seed=0
seed_num=10
log_dir="logs/tuned_vanilla/nifa_degree_eo-both/"

# Loop over datasets

for dataset in nba german bail pokec_z pokec_n; do
    echo
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
    fi

    if [[ "$dataset" == "pokec_n" ]]; then
        nifa_node=87
        nifa_edge=50
        lambda_dp=0.0
        lambda_eo=0.5
    elif [[ "$dataset" == "pokec_z" ]]; then
        nifa_node=102
        nifa_edge=50
        lambda_dp=0.0
        lambda_eo=1.0
    elif [[ "$dataset" == "bail" ]]; then
        nifa_node=25
        nifa_edge=50
        lambda_dp=0.0
        lambda_eo=5.0
    elif [[ "$dataset" == "nba" ]]; then
        nifa_node=4
        nifa_edge=15
        lambda_dp=0.0
        lambda_eo=1.0
    elif [[ "$dataset" == "german" ]]; then
        nifa_node=10
        nifa_edge=50
        lambda_dp=0.0
        lambda_eo=5.0
    fi

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model "$model" \
        --encoder "$encoder" \
        --dataset "$dataset" \
        --start_seed "$start_seed" \
        --seed_num "$seed_num" \
        --epochs "$epochs" \
        --best_overall_path "$best_path" \
        --lambda_dp "$lambda_dp" \
        --lambda_eo "$lambda_eo" \
        --eo_mode "$eo_mode" \
        --log_dir "$log_dir" \
        --attack "$attack" \
        --nifa_mode "$nifa_mode" --nifa_node "$nifa_node" --nifa_edge "$nifa_edge" \
        --nifa_alpha 0.01 --nifa_beta 4 --nifa_ratio 0.5
done

echo
echo "✅ All runs finished."
