set -euo pipefail
CUDA_VISIBLE_DEVICES=3

echo "Running vanilla GIN for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs_piplup/optuna/bail/gin/vanilla/auc_f1/20251007-103133_auto75/best_overall.json"
  [pokec_z]="logs_piplup/optuna/pokec_z/gin/vanilla/auc_f1/20251007-163832_auto75/best_overall.json"
  [pokec_n]="logs_piplup/optuna/pokec_n/gin/vanilla/auc_f1/20251007-135923_auto75/best_overall.json"
  [nba]="logs_piplup/optuna/nba/gin/vanilla/auc_f1/20251007-100231_auto75/best_overall.json"
  [german]="logs_piplup/optuna/german/gin/vanilla/auc_f1/20251007-041741_auto75/best_overall.json"
)

# Common args
encoder="gin"
model="vanilla"
epochs=1000
start_seed=0
seed_num=10
lambda_dp=0.0
lambda_eo=0.0
# max_reduce="max"
# lse_tau=0.5
log_dir="logs/tuned_vanilla/no_attack"
# policy_names="same_largest same_smallest same_random"

# Loop over datasets

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
        --lambda_dp "$lambda_dp" \
        --lambda_eo "$lambda_eo" \
        --log_dir "$log_dir" \
        --attack none
done


echo
echo "✅ All runs finished."
