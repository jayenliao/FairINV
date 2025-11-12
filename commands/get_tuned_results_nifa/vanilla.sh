#!/usr/bin/env bash
# =========================================
# Summarize Optuna tuning results for VANILLA model (5 datasets × 5 backbones)
# It locates each study's `best_overall.json` and runs collect_best_test_stats.py
# to aggregate TEST metrics at the VAL-best epoch.
#
# Usage:
#   ./vanilla.sh [ROOT_DIR] [OUTPUT_DIR] [W_DP] [W_EO] [OBJECTIVE] [BALANCED_ON]
#
# Notes:
# - ROOT_DIR defaults to the first existing of: logs/optuna, logs_piplup/optuna
# - Only studies whose path contains "/vanilla/" are considered.
# - Additional attack options can be passed via env vars:
#     UTIL_ON (default: f1), UTIL_MIN (default: unset), LAMBDA_UTIL (default: 1.0)
# - Python binary may be overridden via $PYTHON
# =========================================

set -euo pipefail

ROOT_DIR="${1:-logs/tune_vanilla_nifa}"
OUTPUT_DIR="${2:-analysis/tuned_nifa_results/vanilla}"
W_DP="${3:-1.0}"
W_EO="${4:-1.0}"
OBJECTIVE="${5:-auc_f1}"
BALANCED_ON="${6:-f1}"

MODEL="vanilla"
PYTHON_BIN="${PYTHON:-python}"

# Attack-related (used only when OBJECTIVE starts with 'attack_')
UTIL_ON="${UTIL_ON:-f1}"
LAMBDA_UTIL="${LAMBDA_UTIL:-1.0}"
# UTIL_MIN can be unset meaning "no constraint".
UTIL_MIN="${UTIL_MIN:-0.55}"

# ---------- Known scenarios ----------
DATASETS=(german bail pokec_z pokec_n nba)
BACKBONES=(gcn gat gin sage sgc)

mkdir -p "$OUTPUT_DIR"

ds_regex="$(IFS='|'; echo "${DATASETS[*]}")"
bb_regex="$(IFS='|'; echo "${BACKBONES[*]}")"

echo "Searching under: $ROOT_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo "Model:           $MODEL"
echo "Objective:       $OBJECTIVE | balanced_on=$BALANCED_ON | w_dp=$W_DP | w_eo=$W_EO"
if [[ "${OBJECTIVE}" == attack_* ]]; then
  echo "Attack opts:     util_on=${UTIL_ON} util_min=${UTIL_MIN:-unset} lambda_util=${LAMBDA_UTIL}"
fi
echo

processed=0
skipped=0

# Resolve collector path (try a few common locations)
resolve_collector() {
  local candidates=(
    "analysis/collect_best_test_stats.py"
    "./collect_best_test_stats.py"
    "$(dirname "$0")/../analysis/collect_best_test_stats.py"
    "$(dirname "$0")/collect_best_test_stats.py"
  )
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  echo "collect_best_test_stats.py"  # hope it's on PYTHONPATH/CWD
}

COLLECTOR="$(resolve_collector)"

# Find all best_overall.json under ROOT_DIR that belong to the VANILLA model
# Typical study dir: <root>/<dataset>/<encoder>/vanilla/<objective>/<timestamp>/best_overall.json
mapfile -t BESTS < <(find "$ROOT_DIR" -type f -name "best_overall.json" | grep "/${MODEL}/" | sort)

if [[ ${#BESTS[@]} -eq 0 ]]; then
  echo "No best_overall.json files found for model=${MODEL} under ${ROOT_DIR}"
  exit 1
fi

for json_file in "${BESTS[@]}"; do
  # Extract dataset/backbone
  ds=$(echo "$json_file" | grep -Eo "/(${ds_regex})/" | sed 's|/||g' | head -n1 || true)
  bb=$(echo "$json_file" | grep -Eo "/(${bb_regex})/" | sed 's|/||g' | head -n1 || true)

  if [[ -z "${ds:-}" || -z "${bb:-}" ]]; then
    echo "⚠️  Skip (couldn't parse dataset/backbone): $json_file"
    ((skipped++)) || true
    continue
  fi

  out_path="$OUTPUT_DIR/${ds}__${bb}_best_overall_stats.json"
  echo "➡️  Processing: ds=$ds  bb=$bb"
  echo "    File: $json_file"

  # Build args
  args=("$json_file" "--objective" "$OBJECTIVE" "--balanced_on" "$BALANCED_ON" "--w_dp" "$W_DP" "--w_eo" "$W_EO" "--out" "$out_path")
  if [[ "${OBJECTIVE}" == attack_* ]]; then
    args+=("--util_on" "$UTIL_ON" "--lambda_util" "$LAMBDA_UTIL")
    if [[ -n "${UTIL_MIN}" ]]; then
      args+=("--util_min" "$UTIL_MIN")
    fi
  fi

  "$PYTHON_BIN" "$COLLECTOR" "${args[@]}"
  ((processed++)) || true
done

echo
echo "✅ Done. Processed: $processed  |  Skipped: $skipped"
expected=$(( ${#DATASETS[@]} * ${#BACKBONES[@]} ))
if [[ $processed -ne $expected ]]; then
  echo "⚠️  Note: expected $expected scenarios but processed $processed. Check your directory layout or names."
fi
echo "Results saved in: $OUTPUT_DIR"
