#!/bin/bash
set -euo pipefail

# ---- Fixed params ----
objective="auc_f1_mean_minus_std"
balanced_on="f1"

# ---- Inputs ----
root_dir=${1:-"logs/optuna"}                         # search root
output_dir=${2:-"analysis/tuned_results"}            # base out dir
w_dp=${3:-1.0}
w_eo=${4:-1.0}
model=${5:-"edge_adder"}                             # <-- choose: edge_adder | fairinv

# Validate model
case "$model" in
  edge_adder|fairinv) ;;
  *) echo "Invalid model: $model (use edge_adder or fairinv)"; exit 1;;
esac

# Scenarios
datasets=(german bail pokec_z pokec_n nba)
backbones=(gcn gat gin sage sgc)

mkdir -p "$output_dir/$model"

ds_regex="$(IFS='|'; echo "${datasets[*]}")"
bb_regex="$(IFS='|'; echo "${backbones[*]}")"

echo "Searching under: $root_dir (model=$model)"
echo "Output dir:      $output_dir/$model"
echo

processed=0
skipped=0

# Only traverse the chosen model subtree:
# logs/optuna/<dataset>/<backbone>/<model>/<objective>/<timestamp>/best_overall.json
while IFS= read -r json_file; do
  ds=$(echo "$json_file" | grep -Eo "/($ds_regex)/" | tr -d '/' | head -n1 || true)
  bb=$(echo "$json_file" | grep -Eo "/($bb_regex)/" | tr -d '/' | head -n1 || true)

  if [[ -z "${ds:-}" || -z "${bb:-}" ]]; then
    echo "⚠️  Skip (couldn't parse dataset/backbone): $json_file"
    ((skipped++)) || true
    continue
  fi

  out_path="$output_dir/$model/${ds}__${bb}__${model}_best_overall_stats.json"

  # Optional: skip if output exists
  # if [[ -f "$out_path" ]]; then echo "↩️  Exists, skip: $out_path"; ((skipped++)) || true; continue; fi

  echo "\n➡️  Processing: ds=$ds  bb=$bb  model=$model"
  python analysis/collect_best_test_stats.py "$json_file" \
    --model "$model" \
    --objective "$objective" \
    --balanced_on "$balanced_on" \
    --w_dp "$w_dp" \
    --w_eo "$w_eo" \
    --out "$out_path"

  ((processed++)) || true
done < <(find "$root_dir" -type f -path "*/$model/*/best_overall.json" -name "best_overall.json" | sort)

echo
echo "✅ Done. Processed: $processed | Skipped: $skipped"
[[ $processed -ne 25 ]] && echo "⚠️ Expected 25 for model=$model; got $processed."

