#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Compare: Vanilla vs Vanilla + Fairness-only AdvTrain
# Eval logs:
#   - test_clean (clean graph)
#   - test       (attacked graph under --attack / --attack_when, typically NIFA at eval)
#
# This script is tailored for the new setting where:
#   - clean BCE is optimized only in the outer minimization
#   - adversarial training acts only on fairness terms
#   - advtrain_attack=edge_weight is the intended default
#
# Hyperparams:
#   - Victim tuned HP from best_overall_json/optuna_big/<model>/<enc>/<ds>.json
#   - Eval-time NIFA HP from best_overall_json/optuna_nifa_expB/<model>_expB/<enc>/<ds>.json
# =========================================================

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

# --------------- User knobs (override via env vars) ---------------
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

# Compare settings
DATASETS="${DATASETS:-bail pokec_z pokec_n german nba}"
ENCODERS="${ENCODERS:-gcn}"

# Eval attack controls.
# Keep ATTACK_WHEN=eval if you want both clean and attacked test metrics.
ATTACK_WHEN="${ATTACK_WHEN:-eval}"   # eval|train|both
EVAL_ATTACK="${EVAL_ATTACK:-nifa}"   # nifa | none | ... whatever train.py supports
SEED_NUM="${SEED_NUM:-5}"
START_SEED="${START_SEED:-42}"
EPOCHS="${EPOCHS:-200}"

# AdvTrain knobs (only applied if train.py supports --advtrain)
# For the fairness-only setting, MIX is usually the safer default than ROBUST.
ADV_MODE="${ADV_MODE:-mix}"          # mix|robust
ADV_K="${ADV_K:-3}"                  # for edge_weight: PGD restarts
ADV_REDUCE="${ADV_REDUCE:-max}"      # mean|max|logsumexp (used when ADV_MODE=robust)
ADV_TAU="${ADV_TAU:-0.5}"
ADV_MIX_LAMBDA="${ADV_MIX_LAMBDA:-1.0}"
ADV_INCLUDE_CLEAN="${ADV_INCLUDE_CLEAN:-1}"   # only used when ADV_MODE=robust

# Fairness coefficients.
# IMPORTANT: in the new setting, the adversarial branch only sees fairness,
# so at least one of these should be > 0.
LAMBDA_DP="${LAMBDA_DP:-1.0}"
LAMBDA_EO="${LAMBDA_EO:-1.0}"

# New: advtrain attack selection
ADVTRAIN_ATTACK="${ADVTRAIN_ATTACK:-edge_weight}"   # edge_weight | nifa | none

# Edge-weight adversary settings (used when ADVTRAIN_ATTACK=edge_weight)
# global_random is usually the cleanest default for the new policy.
ADV_EDGE_POLICY="${ADV_EDGE_POLICY:-global_random}"
ADV_EDGE_K="${ADV_EDGE_K:-4}"               # 0 => fallback to --edge_k (if exists)
ADV_EDGE_STEPS="${ADV_EDGE_STEPS:-5}"
ADV_EDGE_STEP_SIZE="${ADV_EDGE_STEP_SIZE:-0.1}"
ADV_EDGE_GRAD="${ADV_EDGE_GRAD:-sign}"      # sign | raw
ADV_EDGE_W_MAX="${ADV_EDGE_W_MAX:-1.0}"
ADV_EDGE_BUDGET="${ADV_EDGE_BUDGET:--1}"    # <0 disables budget; default -1
EDGE_K="${EDGE_K:-2}"                       # used if ADV_EDGE_K=0 and train.py has --edge_k

# NIFA-specific training knob.
# If ADVTRAIN_ATTACK=nifa and you want fairness-only attack generation too,
# set gamma=0.
ADV_NIFA_GAMMA="${ADV_NIFA_GAMMA:-0}"

# Logging
LOG_TAG="${LOG_TAG:-${ADVTRAIN_ATTACK}_mode-${ADV_MODE}_k-${ADV_K}_dp-${LAMBDA_DP}_eo-${LAMBDA_EO}_policy-${ADV_EDGE_POLICY}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/advtrain_edge_fairness_only/${LOG_TAG}}"
mkdir -p "${LOG_ROOT}"
echo "[INFO] Logs will be written to: ${LOG_ROOT}"
tStart=$(date +%s)
if date -d "@${tStart}" '+%F %T %Z' >/dev/null 2>&1; then
  echo "tStart=${tStart} ($(date -d "@${tStart}" '+%F %T %Z'))"
else
  echo "tStart=${tStart} ($(date -r "${tStart}" '+%F %T %Z'))"
fi

# Behavior
STRICT="${STRICT:-0}"    # 1 => error if any JSON missing; 0 => skip missing combos
DRY_RUN="${DRY_RUN:-0}"  # 1 => only print commands
RUN_BASELINE="${RUN_BASELINE:-0}"  # 1 => also run plain vanilla baseline

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

check_fairness_lambdas() {
  local dp="$1"
  local eo="$2"
  python - "$dp" "$eo" <<'PY'
import sys
try:
    dp = float(sys.argv[1])
    eo = float(sys.argv[2])
except Exception:
    sys.exit(2)
if abs(dp) <= 1e-12 and abs(eo) <= 1e-12:
    sys.exit(1)
sys.exit(0)
PY
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
  echo "eval_attack_hp: ${apath}"
  echo "log   : ${LOG_ROOT}/${tag}"
  echo "=================================================="

  local cmd=(python train.py
    --dataset "${ds}"
    --encoder "${enc}"
    --model "${model}"
    --num_threads "${OMP_NUM_THREADS}"
    --attack "${EVAL_ATTACK}"
    --attack_when "${ATTACK_WHEN}"
    --seed_num "${SEED_NUM}"
    --start_seed "${START_SEED}"
    --epochs "${EPOCHS}"
    --log_dir "${LOG_ROOT}/${tag}"
    --best_overall_path "${vpath}" "${apath}"
    --lambda_dp "${LAMBDA_DP}"
    --lambda_eo "${LAMBDA_EO}"
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
  echo "[WARN] train.py does not expose --advtrain; AdvTrain runs will be skipped."
fi

EDGE_WEIGHT_SUPPORTED=0
if has_flag "--advtrain_attack"; then
  EDGE_WEIGHT_SUPPORTED=1
fi

EDGE_K_SUPPORTED=0
if has_flag "--edge_k"; then
  EDGE_K_SUPPORTED=1
fi

if [[ "${ADVTRAIN_ATTACK}" == "edge_weight" ]]; then
  if ! check_fairness_lambdas "${LAMBDA_DP}" "${LAMBDA_EO}"; then
    echo "[ERROR] Fairness-only edge_weight advtrain needs LAMBDA_DP>0 or LAMBDA_EO>0." >&2
    exit 1
  fi
fi

for ds in ${DATASETS}; do
  for enc in ${ENCODERS}; do

    v_van="best_overall_json/optuna_big/vanilla/${enc}/${ds}.json"
    a_van="best_overall_json/optuna_nifa_expB/vanilla_expB/${enc}/${ds}.json"

    if ! need_json_or_skip "${v_van}"; then
      continue
    fi
    if [[ "${EVAL_ATTACK}" != "none" ]]; then
      if ! need_json_or_skip "${a_van}"; then
        continue
      fi
    else
      a_van="${v_van}"
    fi

    # ---------- Vanilla baseline ----------
    if [[ "${RUN_BASELINE}" == "1" ]]; then
      run_one "${ds}" "${enc}" "vanilla" "vanilla" "${v_van}" "${a_van}"
    fi

    # ---------- Vanilla + AdvTrain ----------
    if [[ "${ADV_SUPPORTED}" == "1" ]]; then
      adv_extra=(--advtrain --advtrain_k "${ADV_K}" --advtrain_mode "${ADV_MODE}")

      # attack selector (if supported)
      if [[ "${EDGE_WEIGHT_SUPPORTED}" == "1" ]]; then
        adv_extra+=(--advtrain_attack "${ADVTRAIN_ATTACK}")
      fi

      # aggregation
      if [[ "${ADV_MODE}" == "mix" ]]; then
        adv_extra+=(--advtrain_mix_lambda "${ADV_MIX_LAMBDA}")
      else
        adv_extra+=(--advtrain_reduce "${ADV_REDUCE}" --advtrain_tau "${ADV_TAU}")
        if [[ "${ADV_INCLUDE_CLEAN}" == "1" ]]; then
          adv_extra+=(--advtrain_include_clean)
        fi
      fi

      # attack-specific knobs
      if [[ "${ADVTRAIN_ATTACK}" == "edge_weight" ]]; then
        adv_extra+=(
          --advtrain_edge_policy "${ADV_EDGE_POLICY}"
          --advtrain_edge_steps "${ADV_EDGE_STEPS}"
          --advtrain_edge_step_size "${ADV_EDGE_STEP_SIZE}"
          --advtrain_edge_grad "${ADV_EDGE_GRAD}"
          --advtrain_edge_w_max "${ADV_EDGE_W_MAX}"
          --advtrain_edge_budget "${ADV_EDGE_BUDGET}"
          --advtrain_edge_k "${ADV_EDGE_K}"
        )
        if [[ "${ADV_EDGE_K}" == "0" && "${EDGE_K_SUPPORTED}" == "1" ]]; then
          adv_extra+=(--edge_k "${EDGE_K}")
        fi
      elif [[ "${ADVTRAIN_ATTACK}" == "nifa" ]]; then
        if has_flag "--advtrain_nifa_gamma"; then
          adv_extra+=(--advtrain_nifa_gamma "${ADV_NIFA_GAMMA}")
        fi
      fi

      run_one "${ds}" "${enc}" "vanilla" "vanilla_advtrain_${ADVTRAIN_ATTACK}" "${v_van}" "${a_van}" "${adv_extra[@]}"
    fi

  done
done

echo
echo "[DONE] Logs written under: ${LOG_ROOT}"
tEnd=$(date +%s)
if date -d "@${tEnd}" '+%F %T %Z' >/dev/null 2>&1; then
  echo "tEnd=${tEnd}   ($(date -d "@${tEnd}" '+%F %T %Z'))"
else
  echo "tEnd=${tEnd}   ($(date -r "${tEnd}" '+%F %T %Z'))"
fi
echo "elapsed=$((tEnd - tStart))s"
