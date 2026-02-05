#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

DATASET="${DATASET:-bail}"
ENCODER="${ENCODER:-gcn}"
MODEL="${MODEL:-vanilla}"
ATTACK_WHEN="${ATTACK_WHEN:-eval}"

if ! has_flag "--advtrain"; then
  echo "[WARN] Your train.py does not expose --advtrain flags. This script is a no-op."
  echo "       If you added the adv-train patch, re-run from that branch."
  exit 0
fi

# Mix-mode adv training
echo "[ADVTRAIN] mix-mode"
extra_mix=()
read -r -a tmp <<<"$(maybe_advtrain_flags mix 2 mean 0.5 1.0)"
extra_mix+=("${tmp[@]}")
run_train "${DATASET}" "${ENCODER}" "${MODEL}" "nifa" "${ATTACK_WHEN}" "${extra_mix[@]}"

# Robust-mode adv training (max)
echo "[ADVTRAIN] robust-mode (max)"
extra_rb=()
read -r -a tmp2 <<<"$(maybe_advtrain_flags robust 2 max 0.5 1.0)"
extra_rb+=("${tmp2[@]}")
run_train "${DATASET}" "${ENCODER}" "${MODEL}" "nifa" "${ATTACK_WHEN}" "${extra_rb[@]}"
