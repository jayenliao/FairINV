#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Compare: EdgeAdder (no advtrain) vs Vanilla vs Vanilla+AdvTrain
# Threat model: Exp-B style (attack_when=eval) by default.
# Hyperparams:
#   - Victim tuned HP from best_overall_json/optuna_big/<model>/<enc>/<ds>.json
#   - Tuned NIFA HP  from best_overall_json/optuna_nifa_expB/<model>_expB/<enc>/<ds>.json
# =========================================================

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

# --------------- User knobs (override via env vars) ---------------
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Compare settings
DATASETS="${DATASETS:-bail pokec_z pokec_n german nba}"
ENCODERS="${ENCODERS:-gcn}"

ATTACK_WHEN="${ATTACK_WHEN:-eval}"   # eval|train|both (for comparison, eval is most fair)
SEED_NUM="${SEED_NUM:-5}"
START_SEED="${START_SEED:-42}"
EPOCHS="${EPOCHS:-200}"

# Logging
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/compare_edgeadder_vs_advtrain_expB}"
mkdir -p "${LOG_ROOT}"

# EdgeAdder knobs (only applied to edge_adder)
EDGE_PIPELINE="${EDGE_PIPELINE:-freeze_gnn_then_edge}"  # joint|freeze_gnn_then_edge
ALT_ROUNDS="${ALT_ROUNDS:-10}"
ALT_EDGE_EPOCHS="${ALT_EDGE_EPOCHS:-20}"
ALT_GNN_EPOCHS="${ALT_GNN_EPOCHS:-20}"

# AdvTrain knobs (only applied if train.py supports --advtrain)
ADV_MODE="${ADV_MODE:-robust}"        # mix|robust
ADV_K="${ADV_K:-2}"
ADV_REDUCE="${ADV_REDUCE:-max}"       # mean|max|logsumexp
ADV_TAU="${ADV_TAU:-0.5}"
ADV_MIX_LAMBDA="${ADV_MIX_LAMBDA:-1.0}"

# Behavior
STRICT="${STRICT:-0}"    # 1 => error if any JSON missing; 0 => skip missing combos
DRY_RUN="${DRY_RUN:-0}"  # 1 => only print commands

# --------------- Helpers ---------------
has_flag() {
  python train.py -h 2>/dev/null | grep -q -- "$1"
}

need_json_or_skip() {
  local p="$1"
  if [[ -f "${p}" ]]; then return 0; fi
  if [[ "${STRICT}" == "1" ]]; then
    echo "[ERROR] Missing JSON: ${p}" >&2
    exit 1
  else
    echo "[SKIP] Missing JSON: ${p}" >&2
    return 1
  fi
}

run_one() {
  local ds="$1"; local enc="$2"; local model="$3"; local tag="$4"
  local vpath="$5"; local apath="$6"
  shift 6
  local extra=("$@")

  echo
  echo "=================================================="
  echo "ds=${ds} enc=${enc} model=${model} tag=${tag} attack_when=${ATTACK_WHEN}"
  echo "victim: ${vpath}"
  echo "nifa  : ${apath}"
  echo "log   : ${LOG_ROOT}/${tag}"
  echo "=================================================="

  local cmd=(python train.py
    --dataset "${ds}"
    --encoder "${enc}"
    --model "${model}"
    --num_threads "${OMP_NUM_THREADS}"
    --attack nifa
    --attack_when "${ATTACK_WHEN}"
    --seed_num "${SEED_NUM}"
    --start_seed "${START_SEED}"
    --epochs "${EPOCHS}"
    --log_dir "${LOG_ROOT}/${tag}"
    --best_overall_path "${vpath}" "${apath}"
  )
  cmd+=("${extra[@]}")

  echo "[CMD] ${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

# --------------- Main ---------------
ADV_SUPPORTED=0
if has_flag "--advtrain"; then
  ADV_SUPPORTED=1
else
  echo "[WARN] train.py does not expose --advtrain; Vanilla+AdvTrain will be skipped."
fi

for ds in ${DATASETS}; do
  for enc in ${ENCODERS}; do

    # ---------- 1) EdgeAdder (no advtrain) ----------
    v_edge="best_overall_json/optuna_big/edge_adder/${enc}/${ds}.json"
    a_edge="best_overall_json/optuna_nifa_expB/edge_adder_expB/${enc}/${ds}.json"
    if need_json_or_skip "${v_edge}" && need_json_or_skip "${a_edge}"; then
      run_one "${ds}" "${enc}" "edge_adder" "edge_adder" "${v_edge}" "${a_edge}" \
        --edge_pipeline "${EDGE_PIPELINE}" \
        --alt_rounds "${ALT_ROUNDS}" \
        --alt_edge_epochs "${ALT_EDGE_EPOCHS}" \
        --alt_gnn_epochs "${ALT_GNN_EPOCHS}"
    fi

    # ---------- 2) Vanilla baseline ----------
    v_van="best_overall_json/optuna_big/vanilla/${enc}/${ds}.json"
    a_van="best_overall_json/optuna_nifa_expB/vanilla_expB/${enc}/${ds}.json"
    if need_json_or_skip "${v_van}" && need_json_or_skip "${a_van}"; then
      run_one "${ds}" "${enc}" "vanilla" "vanilla" "${v_van}" "${a_van}"
    fi

    # ---------- 3) Vanilla + AdvTrain ----------
    if [[ "${ADV_SUPPORTED}" == "1" ]]; then
      if need_json_or_skip "${v_van}" && need_json_or_skip "${a_van}"; then
        adv_extra=(--advtrain --advtrain_k "${ADV_K}" --advtrain_mode "${ADV_MODE}")
        if [[ "${ADV_MODE}" == "mix" ]]; then
          adv_extra+=(--advtrain_mix_lambda "${ADV_MIX_LAMBDA}")
        else
          adv_extra+=(--advtrain_reduce "${ADV_REDUCE}" --advtrain_tau "${ADV_TAU}" --advtrain_include_clean)
        fi
        run_one "${ds}" "${enc}" "vanilla" "vanilla_advtrain" "${v_van}" "${a_van}" "${adv_extra[@]}"
      fi
    fi

  done
done

echo
echo "[DONE] Logs written under: ${LOG_ROOT}"
