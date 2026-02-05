#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "This meta-script runs a small but representative suite."
echo "Override knobs via env vars (EPOCHS, SEED_NUM, DATASETS, ENCODERS, MODELS, etc.)."
echo
bash "${SCRIPT_DIR}/run_baselines.sh"
bash "${SCRIPT_DIR}/run_attack_when.sh"
bash "${SCRIPT_DIR}/run_defenses.sh"
bash "${SCRIPT_DIR}/run_nifa_sweep.sh"
bash "${SCRIPT_DIR}/run_advtrain.sh"
