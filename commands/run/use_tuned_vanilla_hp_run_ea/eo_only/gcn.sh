set -euo pipefail
CUDA_VISIBLE_DEVICES=1

echo "Running EdgeMinMax (GCN backbone) for 5 datasets × 10 seeds..."
echo

# Define datasets and their tuned best_overall.json paths
declare -A best_paths=(
  [bail]="logs_piplup/optuna/bail/gcn/vanilla/auc_f1/20251006-003953_auto75/best_overall.json"
  [pokec_z]="logs_piplup/optuna/pokec_z/gcn/vanilla/auc_f1/20251006-004128_auto75/best_overall.json"
  [pokec_n]="logs_piplup/optuna/pokec_n/gcn/vanilla/auc_f1/20251006-004151_auto75/best_overall.json"
  [nba]="logs_piplup/optuna/nba/gcn/vanilla/auc_f1/20251006-004228_auto75/best_overall.json"
  [german]="logs_piplup/optuna/german/gcn/vanilla/auc_f1/20251006-003808_auto75/best_overall.json"
)


# Common args
encoder="gcn"
model="edge_adder"
epochs=1000
start_seed=0
seed_num=10
lambda_eo=(0.1 0.5 1.0 5.0 10.0)
lambda_dp=0.0
# max_reduce="max"
# lse_tau=0.5
log_dir=(
  "logs/use_tuned_vanilla_hp_run_ea/dp_0_eo_1e-1"
  "logs/use_tuned_vanilla_hp_run_ea/dp_0_eo_5e-1"
  "logs/use_tuned_vanilla_hp_run_ea/dp_0_eo_1"
  "logs/use_tuned_vanilla_hp_run_ea/dp_0_eo_5"
  "logs/use_tuned_vanilla_hp_run_ea/dp_0_eo_10"
)
# policy_names="same_largest same_smallest same_random"

# Loop over datasets
for i in "${!lambda_eo[@]}"; do
  lambda_eo_val="${lambda_eo[$i]}"
  log_dir_val="${log_dir[$i]}"

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
      --lambda_eo "$lambda_eo_val" \
      --log_dir "$log_dir_val"
  done
done

echo
echo "✅ All runs finished."
