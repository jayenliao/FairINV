#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

DATASETS="${DATASETS:-german bail pokec_z pokec_n nba}"
ENCODERS="${ENCODERS:-gcn}"
ATTACK_WHEN="${ATTACK_WHEN:-eval}"   # eval-time injection is the cleanest "robustness" test

# Compare models under the same attack setting
MODELS="${MODELS:-vanilla fairinv edge_adder edge_minmax}"

# EdgeAdder knobs you may want to override via env vars
EDGE_PIPELINE="${EDGE_PIPELINE:-joint}"              # joint | freeze_gnn_then_edge
EDGE_CAND_SOURCE="${EDGE_CAND_SOURCE:-feat}"         # feat | emb
EDGE_K="${EDGE_K:-2}"
LAMBDA_EDGE_L1="${LAMBDA_EDGE_L1:-0.0}"
ADV_REDUCE_EXCLUDE_L1="${ADV_REDUCE_EXCLUDE_L1:-0}"
SCALE_LAMBDA="${SCALE_LAMBDA:-0}"
MAX_REDUCE="${MAX_REDUCE:-max}"                      # max | lse
LSE_TAU="${LSE_TAU:-0.5}"

for ds in ${DATASETS}; do
  for enc in ${ENCODERS}; do
    for m in ${MODELS}; do
      extra=()
      if [[ "${m}" == "edge_adder" || "${m}" == "edge_minmax" ]]; then
        extra+=(
          "--edge_pipeline" "${EDGE_PIPELINE}"
          "--edge_cand_source" "${EDGE_CAND_SOURCE}"
          "--edge_k" "${EDGE_K}"
          "--lambda_edge_l1" "${LAMBDA_EDGE_L1}"
          "--max_reduce" "${MAX_REDUCE}"
          "--lse_tau" "${LSE_TAU}"
        )
        if [[ "${ADV_REDUCE_EXCLUDE_L1}" == "1" ]]; then
          extra+=("--adv_reduce_exclude_l1")
        fi
        if [[ "${SCALE_LAMBDA}" == "1" ]]; then
          extra+=("--scale_lambda")
        fi
      fi
      run_train "${ds}" "${enc}" "${m}" "nifa" "${ATTACK_WHEN}" "${extra[@]}"
    done
  done
done
