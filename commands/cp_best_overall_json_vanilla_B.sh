#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash commands/cp_best_overall_json_vanilla.sh <BASE_DIR> [MODEL] [METRIC]
# Example:
#   bash commands/cp_best_overall_json_vanilla.sh logs/optuna_big vanilla auc_f1

BASE_DIR="${1:-logs/optuna_nifa_expB}"
MODEL="${2:-vanilla}"
METRIC="${3:-attack_balanced}"   # default to auc_f1 to avoid overwriting across metrics
EXP="${4:-expB}"

# Output root follows your requested layout:
# best_overall_json/<basename(BASE_DIR)>/<MODEL>/<gnn>/<dataset>.json
OUT_ROOT="best_overall_json/$(basename "${BASE_DIR}")/${MODEL}_${EXP}"
mkdir -p "${OUT_ROOT}"

echo "[INFO] BASE_DIR=${BASE_DIR}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] METRIC=${METRIC}"
echo "[INFO] EXP=${EXP}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo

count=0

# Call your lister, then copy only entries matching /MODEL/METRIC/.../best_overall.json
bash ./commands/list_best_overall.sh "${BASE_DIR}" "${MODEL}" \
  | grep -E "/${MODEL}/${METRIC}/.*/best_overall\.json$" \
  | while read -r src; do
      [[ -z "${src}" ]] && continue

      # src: <BASE_DIR>/<dataset>/<gnn>/<MODEL>/<METRIC>/<run_id>/best_overall.json
      rel="${src#${BASE_DIR}/}"
      IFS='/' read -r dataset gnn model metric _rest <<< "${rel}"

      dest_dir="${OUT_ROOT}/${gnn}"
      dest="${dest_dir}/${dataset}.json"
      mkdir -p "${dest_dir}"

      cp -v "${src}" "${dest}"
      count=$((count+1))
    done

echo
echo "✅ Done. Copied best_overall.json -> ${OUT_ROOT}/{gnn}/{dataset}.json"
echo "   (If you want other metrics, pass METRIC=auc_f1_balanced / attack_balanced / auc_f1_mean_minus_std)"
