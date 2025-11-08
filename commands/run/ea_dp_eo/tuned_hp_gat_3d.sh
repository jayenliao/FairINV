set -euo pipefail
CUDA_VISIBLE_DEVICES=2

echo "Running EdgeMinMax (GCN backbone) for 5 datasets × 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs/optuna/bail/gat/edge_adder/auc_f1_mean_minus_std/20251007-102912_robust/best_overall.json"
  [pokec_z]="logs/optuna/pokec_z/gat/edge_adder/auc_f1_mean_minus_std/20251008-205103_robust/best_overall.json"
  [pokec_n]="logs/optuna/pokec_n/gat/edge_adder/auc_f1_mean_minus_std/20251010-183433_robust/best_overall.json"
  [nba]="logs/optuna/nba/gat/edge_adder/auc_f1/20251006-215936_auto75/best_overall.json"
  [german]="logs/optuna/german/gat/edge_adder/auc_f1/20251006-113632_auto75/best_overall.json"
)

# Common args
encoder="gat"
model="edge_adder"
epochs=1000
start_seed=0
seed_num=10
lambda_eo=-1.0
# max_reduce="max"
# lse_tau=0.5
log_dir="logs/ea_dp_eo"
# policy_names="same_largest same_smallest same_random"

# Loop over datasets
for dataset in bail pokec_z pokec_n; do
  echo
  echo "============ ${dataset^^} ============="
  best_path="${best_paths[$dataset]}"
  if [[ ! -f "$best_path" ]]; then
    echo "⚠️  Warning: best_overall.json not found for $dataset at $best_path"
  fi

  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_eo.py \
    --model "$model" \
    --encoder "$encoder" \
    --dataset "$dataset" \
    --start_seed "$start_seed" \
    --seed_num "$seed_num" \
    --epochs "$epochs" \
    --best_overall_path "$best_path" \
    --lambda_eo "$lambda_eo" \
    --log_dir "$log_dir"
done

echo
echo "✅ All runs finished."
