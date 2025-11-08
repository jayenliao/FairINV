set -euo pipefail
CUDA_VISIBLE_DEVICES=1

echo "Attacking FairINV with GIN backbone for 5 datasets x 10 seeds..."
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
log_dir="logs/tuned_fairinv/nifa_uncertainty"
attack="nifa"
nifa_mode='uncertainty'
# max_reduce="max"
# lse_tau=0.5
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
        --attack "$attack" \
        --nifa_mode "$nifa_mode" --nifa_node "$nifa_node" --nifa_edge "$nifa_edge" \
        --nifa_alpha 0.01 --nifa_beta 4 --nifa_ratio 0.5
done


echo
echo "✅ All runs finished."
