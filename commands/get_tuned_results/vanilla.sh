#!/bin/bash
# =========================================
# Collect best_overall stats for 25 scenarios
# Datasets: german, bail, pokec_z, pokec_n, nba
# Backbones: gcn, gat, gin, sage, sgc
# =========================================

set -euo pipefail

# ---- Configurable parameters ----
objective="auc_f1"
balanced_on="f1"
model="vanilla"

# ---- Inputs ----
root_dir=${1:-"logs_piplup/optuna"}              # Search here for best_overall.json
output_dir=${2:-"analysis/tuning_results/vanilla"}               # Where to write stats JSONs
w_dp=${3:-1.0}
w_eo=${4:-1.0}

# ---- Known scenarios ----
datasets=(german bail pokec_z pokec_n nba)
backbones=(gcn gat gin sage sgc)

# ---- Prep ----
mkdir -p "$output_dir"

# Build regex to match only the 5x5 scenarios
ds_regex="$(IFS='|'; echo "${datasets[*]}")"
bb_regex="$(IFS='|'; echo "${backbones[*]}")"

echo "Searching under: $root_dir"
echo "Output dir:      $output_dir"
echo "Objective:       $objective | balanced_on=$balanced_on | w_dp=$w_dp | w_eo=$w_eo"
echo

processed=0
skipped=0

# Find all best_overall.json files (case-sensitive)
while IFS= read -r json_file; do
  # Try to extract dataset/backbone from the path (matches first occurrence)
  ds=$(echo "$json_file" | grep -Eo "/($ds_regex)/" | sed 's|/||g' | head -n1 || true)
  bb=$(echo "$json_file" | grep -Eo "/($bb_regex)/" | sed 's|/||g' | head -n1 || true)

  # Fallback: try filename or neighboring dirs if needed
  if [[ -z "${ds:-}" ]]; then
    ds=$(echo "$json_file" | grep -Eo "($ds_regex)" | head -n1 || true)
  fi
  if [[ -z "${bb:-}" ]]; then
    bb=$(echo "$json_file" | grep -Eo "($bb_regex)" | head -n1 || true)
  fi

  if [[ -z "${ds:-}" || -z "${bb:-}" ]]; then
    echo "⚠️  Skip (couldn't parse dataset/backbone): $json_file"
    ((skipped++)) || true
    continue
  fi

  out_path="$output_dir/${ds}__${bb}_best_overall_stats.json"
  echo "➡️  Processing: ds=$ds  bb=$bb"
  echo "    File: $json_file"
  python analysis/collect_best_test_stats.py "$json_file" \
    --objective "$objective" \
    --balanced_on "$balanced_on" \
    --w_dp "$w_dp" \
    --w_eo "$w_eo" \
    --out "$out_path"
  ((processed++)) || true
done < <(find "$root_dir" -type f -name "best_overall.json" | sort)

echo
echo "✅ Done. Processed: $processed  |  Skipped: $skipped"
if [[ $processed -ne 25 ]]; then
  echo "⚠️  Note: expected 25 scenarios but processed $processed. Check your directory layout or names."
fi
echo "Results saved in: $output_dir"
