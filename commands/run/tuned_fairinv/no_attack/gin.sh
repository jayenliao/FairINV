set -euo pipefail
CUDA_VISIBLE_DEVICES=1

echo "Running FairINV with GIN backbone for 5 datasets x 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs/optuna/bail/gin/fairinv/auc_f1_mean_minus_std/20251007-194534_robust/best_overall.json"
  [pokec_z]="logs/optuna/pokec_z/gin/fairinv/auc_f1_mean_minus_std/20251008-103142_robust/best_overall.json"
  [pokec_n]="logs/optuna/pokec_n/gin/fairinv/auc_f1_mean_minus_std/20251009-031344_robust/best_overall.json"
  [nba]="logs/optuna/nba/gin/fairinv/auc_f1_mean_minus_std/20251007-070131_robust/best_overall.json"
  [german]="logs/optuna/german/gin/fairinv/auc_f1_mean_minus_std/20251006-223953_robust/best_overall.json"
)

# Common args
encoder="gin"
model="fairinv"
epochs=500
start_seed=0
seed_num=10
lambda_dp=0.0
lambda_eo=0.0
# max_reduce="max"
# lse_tau=0.5
log_dir="logs/tuned_fairinv/no_attack"
# policy_names="same_largest same_smallest same_random"

for dataset in bail pokec_z pokec_n nba german; do
    echo
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    if [[ ! -f "$best_path" ]]; then
        echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
        exit 1
    else
        echo "Using best_overall.json at: $best_path"
    fi
done

for dataset in bail pokec_z pokec_n nba german; do
    echo
    echo "============ ${dataset^^} ============="
    best_path="${best_paths[$dataset]}"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --num_threads 16 \
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
