#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=3

DRY_RUN=${DRY_RUN:-0}  # 1 = only print commands, 0 = actually run

run_cmd () {
  local cmd="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY_RUN] $cmd"
  else
    eval "$cmd"
  fi
}

echo "Running tuned HPs for 5 datasets x gin x 10 seeds..."
echo "DRY_RUN=$DRY_RUN"
echo

# -----------------------------
# Tuned best_overall.json paths
# (gin templates; we swap encoder)
# -----------------------------
declare -A best_vanilla_paths=(
  [bail]="best_overall_json/vanilla/gin/bail.json"
  [pokec_z]="best_overall_json/vanilla/gin/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/gin/pokec_n.json"
  [nba]="best_overall_json/vanilla/gin/nba.json"
  [german]="best_overall_json/vanilla/gin/german.json"
)

declare -A best_ea_paths=(
  [bail]="best_overall_json/ea_obj/gin/bail.json"
  [pokec_z]="best_overall_json/ea_obj/gin/pokec_z.json"
  [pokec_n]="best_overall_json/ea_obj/gin/pokec_n.json"
  [nba]="best_overall_json/ea_obj/gin/nba.json"
  [german]="best_overall_json/ea_obj/gin/german.json"
)

# -----------------------------
# Common args
# -----------------------------
attack="none"
epochs=500
start_seed=0
seed_num=10

# Run switches
run_vanilla=1
run_ea=1

# If you only tuned gin and want to reuse it for other backbones:
allow_gin_fallback=1

# Log roots (train.py will append dataset/encoder/model/timestamp)
log_dir_vanilla="logs/scaled_use_tuned_hp/tuned_vanilla/"
log_dir_ea="logs/scaled_use_tuned_hp/tuned_ea/"

# Optional: your new pipeline flags (only applies to EdgeAdder)
edge_pipeline="freeze_gnn_then_edge"   # or "joint"
edge_cand_source="emb"                # or "feat"
# pretrain_epochs=200
# edge_epochs=200

# -----------------------------
# Loops: gin x 5 datasets
# -----------------------------
for encoder in gin; do
  echo
  echo "================ ENCODER: ${encoder^^} ================"

  for dataset in bail pokec_z pokec_n german nba; do
    echo
    echo "------------ ${dataset^^} ------------"

    vanilla_tmpl="${best_vanilla_paths[$dataset]}"
    ea_tmpl="${best_ea_paths[$dataset]}"

    # Swap encoder folder: /gin/ -> /${encoder}/
    best_vanilla_path="${vanilla_tmpl/\/gin\//\/${encoder}\/}"
    best_ea_path="${ea_tmpl/\/gin\//\/${encoder}\/}"

    # Check / fallback (same spirit as your script, but avoids crashing)
    if [[ ! -f "$best_vanilla_path" ]]; then
      echo "⚠️  Warning: vanilla best_overall.json not found at: $best_vanilla_path"
      if [[ "$allow_gin_fallback" -eq 1 && -f "$vanilla_tmpl" ]]; then
        echo "    -> fallback to gin tuned file: $vanilla_tmpl"
        best_vanilla_path="$vanilla_tmpl"
      else
        echo "    -> skip vanilla / EA for this combo (no vanilla tuned file)."
        continue
      fi
    fi

    run_ea_this=1
    if [[ ! -f "$best_ea_path" ]]; then
      echo "⚠️  Warning: ea_obj best_overall.json not found at: $best_ea_path"
      if [[ "$allow_gin_fallback" -eq 1 && -f "$ea_tmpl" ]]; then
        echo "    -> fallback to gin tuned file: $ea_tmpl"
        best_ea_path="$ea_tmpl"
      else
        echo "    -> skip EA for this combo (no EA tuned file)."
        run_ea_this=0
      fi
    fi

    echo "Using tuned vanilla: $best_vanilla_path"
    if [[ "$run_ea_this" -eq 1 ]]; then
      echo "Using tuned ea_obj : $best_ea_path"
    fi

    # ---- Vanilla ----
    if [[ "$run_vanilla" -eq 1 ]]; then
      cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model vanilla \
        --encoder $encoder \
        --dataset $dataset \
        --start_seed $start_seed \
        --seed_num $seed_num \
        --epochs $epochs \
        --best_overall_path $best_vanilla_path \
        --log_dir $log_dir_vanilla \
        --attack $attack"
      run_cmd "$cmd"
    fi

    # ---- EdgeAdder (load vanilla first, then ea_obj) ----
    if [[ "$run_ea" -eq 1 && "$run_ea_this" -eq 1 ]]; then
      cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
        --model edge_adder \
        --encoder $encoder \
        --dataset $dataset \
        --start_seed $start_seed \
        --seed_num $seed_num \
        --epochs $epochs \
        --best_overall_path $best_vanilla_path $best_ea_path \
        --log_dir $log_dir_ea \
        --attack $attack \
        --edge_pipeline $edge_pipeline \
        --edge_cand_source $edge_cand_source"
        # --pretrain_epochs $pretrain_epochs \
        # --edge_epochs $edge_epochs
      run_cmd "$cmd"
    fi

  done
done

echo
echo "✅ All runs finished."
