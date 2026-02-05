#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

# Focused NIFA sensitivity sweep (kept small by default)
DATASET="${DATASET:-bail}"
ENCODER="${ENCODER:-gcn}"
MODEL="${MODEL:-vanilla}"
ATTACK_WHEN="${ATTACK_WHEN:-eval}"

NODES="${NODES:-32 64 128}"
EDGES="${EDGES:-20 50}"
BETAS="${BETAS:-0.5 1.0 2.0}"
RATIOS="${RATIOS:-0.25 0.5}"
MODES="${MODES:-uncertainty degree}"

# If COMBO_MODE=paired, we pair the i-th element across lists (requires same lengths).
COMBO_MODE="${COMBO_MODE:-grid}"   # grid | paired

if [[ "${COMBO_MODE}" == "paired" ]]; then
  # Convert to arrays
  read -r -a a_nodes <<<"${NODES}"
  read -r -a a_edges <<<"${EDGES}"
  read -r -a a_betas <<<"${BETAS}"
  read -r -a a_ratios <<<"${RATIOS}"
  read -r -a a_modes <<<"${MODES}"
  n="${#a_nodes[@]}"
  if [[ "${#a_edges[@]}" -ne "${n}" || "${#a_betas[@]}" -ne "${n}" || "${#a_ratios[@]}" -ne "${n}" || "${#a_modes[@]}" -ne "${n}" ]]; then
    echo "paired mode requires NODES/EDGES/BETAS/RATIOS/MODES have the same length"
    exit 1
  fi
  for i in $(seq 0 $((n-1))); do
    NIFA_NODE="${a_nodes[i]}" NIFA_EDGE="${a_edges[i]}" NIFA_BETA="${a_betas[i]}" NIFA_RATIO="${a_ratios[i]}" NIFA_MODE="${a_modes[i]}" \
      run_train "${DATASET}" "${ENCODER}" "${MODEL}" "nifa" "${ATTACK_WHEN}"
  done
else
  for n in ${NODES}; do
    for e in ${EDGES}; do
      for b in ${BETAS}; do
        for r in ${RATIOS}; do
          for m in ${MODES}; do
            NIFA_NODE="${n}" NIFA_EDGE="${e}" NIFA_BETA="${b}" NIFA_RATIO="${r}" NIFA_MODE="${m}" \
              run_train "${DATASET}" "${ENCODER}" "${MODEL}" "nifa" "${ATTACK_WHEN}"
          done
        done
      done
    done
  done
fi
