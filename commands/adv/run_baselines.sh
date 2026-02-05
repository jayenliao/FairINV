#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

DATASETS="${DATASETS:-german bail pokec_z pokec_n nba}"
ENCODERS="${ENCODERS:-gcn gat gin sage sgc}"
MODELS="${MODELS:-vanilla}"

# Baseline: no attack
for ds in ${DATASETS}; do
  for enc in ${ENCODERS}; do
    for m in ${MODELS}; do
      run_train "${ds}" "${enc}" "${m}" "none" "train"
    done
  done
done
