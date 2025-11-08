set -euo pipefail
CUDA_VISIBLE_DEVICES=3

echo "Running EdgeAdder with SGC backbone for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs_piplup/optuna/bail/sgc/vanilla/auc_f1/20251007-190642_auto75/best_overall.json"
  [pokec_z]="logs_piplup/optuna/pokec_z/sgc/vanilla/auc_f1/20251007-234448_auto75/best_overall.json"
  [pokec_n]="logs_piplup/optuna/pokec_n/sgc/vanilla/auc_f1/20251007-221300_auto75/best_overall.json"
  [nba]="logs_piplup/optuna/nba/sgc/vanilla/auc_f1/20251007-174119_auto75/best_overall.json"
  [german]="logs_piplup/optuna/german/sgc/vanilla/auc_f1/20251007-173256_auto75/best_overall.json"
)

# Common args
encoder="sgc"
model="edge_adder"
attack="none"
epochs=1000
start_seed=0
seed_num=10
lambda_dp=0.0
lambda_eo=0.0
log_dir="logs/tuned_vanilla/no_attack"

# Loop over datasets

for dataset in bail pokec_z pokec_n german nba; do
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
    fi

    if [[ "$dataset" == "pokec_n" ]]; then
        lambda_dp=10.0
        lambda_eo=0.0
    elif [[ "$dataset" == "pokec_z" ]]; then
        lambda_dp=1.0
        lambda_eo=0.0
    elif [[ "$dataset" == "bail" ]]; then
        lambda_dp=0.0
        lambda_eo=5.0
    elif [[ "$dataset" == "nba" ]]; then
        lambda_dp=10.0
        lambda_eo=0.0
    elif [[ "$dataset" == "german" ]]; then
        lambda_dp=1.0
        lambda_eo=0.0
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
