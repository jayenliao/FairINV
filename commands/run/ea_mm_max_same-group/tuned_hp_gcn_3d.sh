set -euo pipefail
CUDA_VISIBLE_DEVICES=1

echo "Running EdgeMinMax with Min-Max game (GCN backbone) for 5 datasets × 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs/optuna/bail/gcn/edge_adder/auc_f1_mean_minus_std/20251007-102858_robust/best_overall.json"
  [pokec_z]="logs/optuna/pokec_z/gcn/edge_adder/auc_f1_mean_minus_std/20251008-140347_robust/best_overall.json"
  [pokec_n]="logs/optuna/pokec_n/gcn/edge_adder/auc_f1_mean_minus_std/20251009-121513_robust/best_overall.json"
  [nba]="logs/optuna/nba/gcn/edge_adder/auc_f1/20251006-215512_auto75/best_overall.json"
  [german]="logs/optuna/german/gcn/edge_adder/auc_f1/20251006-113530_auto75/best_overall.json"
)

# Common args
encoder="gcn"
model="edge_minmax"
epochs=1000
start_seed=0
seed_num=10
max_reduce="max"
lse_tau=0.5
log_dir="logs/ea_mm_max_same-group"
policy_names="same_largest same_smallest same_random"

# Loop over datasets
for dataset in bail pokec_z pokec_n; do
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
    --max_reduce "$max_reduce" \
    --lse_tau "$lse_tau" \
    --best_overall_path "$best_path" \
    --policy_names $policy_names \
    --log_dir "$log_dir"
done

echo
echo "✅ All runs finished."
