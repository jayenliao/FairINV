set -euo pipefail
CUDA_VISIBLE_DEVICES=1

echo "Running EdgeMinMax (GraphSAGE backbone) for 5 datasets × 10 seeds..."
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
epochs=1000
start_seed=0
seed_num=10
lambda_dp=(0.1 0.5 1.0 5.0 10.0)
lambda_eo=0.0
# max_reduce="max"
# lse_tau=0.5
log_dir=(
  "logs/use_tuned_vanilla_hp_run_ea/dp_1e-1_eo_0"
  "logs/use_tuned_vanilla_hp_run_ea/dp_5e-1_eo_0"
  "logs/use_tuned_vanilla_hp_run_ea/dp_1_eo_0"
  "logs/use_tuned_vanilla_hp_run_ea/dp_5_eo_0"
  "logs/use_tuned_vanilla_hp_run_ea/dp_10_eo_0"
)
# policy_names="same_largest same_smallest same_random"

# Loop over datasets
for i in "${!lambda_dp[@]}"; do
  lambda_dp_val="${lambda_dp[$i]}"
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
      --lambda_dp "$lambda_dp_val" \
      --lambda_eo "$lambda_eo" \
      --log_dir "$log_dir_val"
  done
done

echo
echo "✅ All runs finished."
