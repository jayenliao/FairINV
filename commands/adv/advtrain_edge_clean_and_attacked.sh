#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Compare: Vanilla vs Vanilla + AdvTrain (Exp-B: attack_when=eval)
# This version supports advtrain_attack=edge_weight, and will log:
#   - test_clean (clean graph)
#   - test       (attacked graph, nifa, when ATTACK_WHEN=eval)
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

ATTACK_WHEN="${ATTACK_WHEN:-eval}"   # eval|train|both (eval => logs clean+attacked if train.py supports it)
SEED_NUM="${SEED_NUM:-5}"
START_SEED="${START_SEED:-42}"
EPOCHS="${EPOCHS:-200}"

# AdvTrain knobs (only applied if train.py supports --advtrain)
ADV_MODE="${ADV_MODE:-robust}"        # mix|robust
ADV_K="${ADV_K:-2}"                   # for edge_weight: PGD restarts
ADV_REDUCE="${ADV_REDUCE:-max}"       # mean|max|logsumexp
ADV_TAU="${ADV_TAU:-0.5}"
ADV_MIX_LAMBDA="${ADV_MIX_LAMBDA:-1.0}"

# Choice (1): "increase fairness weight in adversarial objective"
# In the current implementation, inner PGD maximizes (task + lambda_dp*DP + lambda_eo*EO),
# so setting these >0 strengthens fairness inside the adversary too.
LAMBDA_DP="${LAMBDA_DP:-1.0}"
LAMBDA_EO="${LAMBDA_EO:-1.0}"

# New: advtrain attack selection
ADVTRAIN_ATTACK="${ADVTRAIN_ATTACK:-edge_weight}"   # edge_weight | nifa | none

# Edge-weight adversary settings (used when ADVTRAIN_ATTACK=edge_weight)
ADV_EDGE_POLICY="${ADV_EDGE_POLICY:-same_smallest}"
ADV_EDGE_K="${ADV_EDGE_K:-0}"               # 0 => fallback to --edge_k (if exists)
ADV_EDGE_STEPS="${ADV_EDGE_STEPS:-5}"
ADV_EDGE_STEP_SIZE="${ADV_EDGE_STEP_SIZE:-0.1}"
ADV_EDGE_GRAD="${ADV_EDGE_GRAD:-sign}"      # sign | raw
ADV_EDGE_W_MAX="${ADV_EDGE_W_MAX:-1.0}"
ADV_EDGE_BUDGET="${ADV_EDGE_BUDGET:-1}"     # <0 disables budget; default -1
EDGE_K="${EDGE_K:-2}"                       # used if ADV_EDGE_K=0 and train.py has --edge_k

# Logging
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/advtrain_edge/${ADVTRAIN_ATTACK}/${ADV_EDGE_POLICY}/k-${ADV_K}_steps-${ADV_EDGE_STEPS}_dp-${LAMBDA_DP}_eo-${LAMBDA_EO}}"
mkdir -p "${LOG_ROOT}"
echo "[INFO] Logs will be written to: ${LOG_ROOT}"
tStart=$(date +%s)
echo "tStart=$tStart ($(date -d "@$tStart" '+%F %T %Z'))"

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

for ds in ${DATASETS}; do
  for enc in ${ENCODERS}; do

    # ---------- Vanilla baseline (optional; uncomment if you want it) ----------
    # v_van="best_overall_json/optuna_big/vanilla/${enc}/${ds}.json"
    # a_van="best_overall_json/optuna_nifa_expB/vanilla_expB/${enc}/${ds}.json"
    # if need_json_or_skip "${v_van}" && need_json_or_skip "${a_van}"; then
    #   run_one "${ds}" "${enc}" "vanilla" "vanilla" "${v_van}" "${a_van}"
    # fi

    # ---------- Vanilla + AdvTrain ----------
    if [[ "${ADV_SUPPORTED}" == "1" ]]; then
      v_van="best_overall_json/optuna_big/vanilla/${enc}/${ds}.json"
      a_van="best_overall_json/optuna_nifa_expB/vanilla_expB/${enc}/${ds}.json"
      if need_json_or_skip "${v_van}" && need_json_or_skip "${a_van}"; then

        adv_extra=(--advtrain --advtrain_k "${ADV_K}" --advtrain_mode "${ADV_MODE}")

        # attack selector (if supported)
        if [[ "${EDGE_WEIGHT_SUPPORTED}" == "1" ]]; then
          adv_extra+=(--advtrain_attack "${ADVTRAIN_ATTACK}")
        fi

        # robust aggregation
        if [[ "${ADV_MODE}" == "mix" ]]; then
          adv_extra+=(--advtrain_mix_lambda "${ADV_MIX_LAMBDA}")
        else
          adv_extra+=(--advtrain_reduce "${ADV_REDUCE}" --advtrain_tau "${ADV_TAU}" --advtrain_include_clean)
        fi

        # edge_weight-specific knobs
        if [[ "${ADVTRAIN_ATTACK}" == "edge_weight" ]]; then
          adv_extra+=(
            --advtrain_edge_policy "${ADV_EDGE_POLICY}"
            --advtrain_edge_steps "${ADV_EDGE_STEPS}"
            --advtrain_edge_step_size "${ADV_EDGE_STEP_SIZE}"
            --advtrain_edge_grad "${ADV_EDGE_GRAD}"
            --advtrain_edge_w_max "${ADV_EDGE_W_MAX}"
            --advtrain_edge_budget "${ADV_EDGE_BUDGET}"
          )
          # candidate density
          adv_extra+=(--advtrain_edge_k "${ADV_EDGE_K}")
          # fallback edge_k (only if supported)
          if [[ "${ADV_EDGE_K}" == "0" && "${EDGE_K_SUPPORTED}" == "1" ]]; then
            adv_extra+=(--edge_k "${EDGE_K}")
          fi
        fi

        run_one "${ds}" "${enc}" "vanilla" "vanilla_advtrain_${ADVTRAIN_ATTACK}" "${v_van}" "${a_van}" "${adv_extra[@]}"
      fi
    fi

  done
done

echo
echo "[DONE] Logs written under: ${LOG_ROOT}"
tEnd=$(date +%s)
echo "tEnd=$tEnd   ($(date -d "@$tEnd" '+%F %T %Z'))"
echo "elapsed=$((tEnd - tStart))s"
