#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

DATASETS="${DATASETS:-german bail pokec_z pokec_n nba}"
ENCODERS="${ENCODERS:-gcn}"
MODELS="${MODELS:-vanilla}"

ATTACK_WHENS="${ATTACK_WHENS:-train eval both}"

for ds in ${DATASETS}; do
  for enc in ${ENCODERS}; do
    for m in ${MODELS}; do
      for aw in ${ATTACK_WHENS}; do
        run_train "${ds}" "${enc}" "${m}" "nifa" "${aw}"
      done
    done
  done
done
