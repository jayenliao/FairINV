set -euo pipefail
CUDA_VISIBLE_DEVICES=3

echo "Running EdgeAdder with GraphSAGE backbone for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs_piplup/optuna/bail/sage/vanilla/auc_f1/20251007-144828_auto75/best_overall.json"
  [pokec_z]="logs_piplup/optuna/pokec_z/sage/vanilla/auc_f1/20251007-195439_auto75/best_overall.json"
  [pokec_n]="logs_piplup/optuna/pokec_n/sage/vanilla/auc_f1/20251007-181427_auto75/best_overall.json"
  [nba]="logs_piplup/optuna/nba/sage/vanilla/auc_f1/20251007-140649_auto75/best_overall.json"
  [german]="logs_piplup/optuna/german/sage/vanilla/auc_f1/20251007-135712_auto75/best_overall.json"
)

# Common args
encoder="sage"
model="edge_adder"
attack="none"
epochs=1000
start_seed=0
seed_num=10
log_dir="logs/tuned_vanilla/no_attack"

# Loop over datasets

for dataset in bail pokec_z pokec_n german nba; do
    echo
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
    fi

    if [[ "$dataset" == "pokec_n" ]]; then
        lambda_dp=0.0
        lambda_eo=0.1
    elif [[ "$dataset" == "pokec_z" ]]; then
        lambda_dp=5.0
        lambda_eo=0.0
    elif [[ "$dataset" == "bail" ]]; then
        lambda_dp=0.0
        lambda_eo=10.0
    elif [[ "$dataset" == "nba" ]]; then
        lambda_dp=5.0
        lambda_eo=0.0
    elif [[ "$dataset" == "german" ]]; then
        lambda_dp=0.0
        lambda_eo=10.0
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
        --log_dir "$log_dir" \
        --attack "$attack"
done

echo
echo "✅ All runs finished."
