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

echo "Running EdgeAdder under NIFA (uncertainty) for 5 datasets x 1 backbone (sgc) x 10 seeds..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DRY_RUN=$DRY_RUN"
echo

# ---------------------------------------------
# Tuned best_overall.json paths (sgc templates)
# We swap /sgc/ -> /${encoder}/
# ---------------------------------------------
declare -A best_vanilla_paths=(
  [bail]="best_overall_json/vanilla/sgc/bail.json"
  [pokec_z]="best_overall_json/vanilla/sgc/pokec_z.json"
  [pokec_n]="best_overall_json/vanilla/sgc/pokec_n.json"
  [nba]="best_overall_json/vanilla/sgc/nba.json"
  [german]="best_overall_json/vanilla/sgc/german.json"
)

# declare -A best_nifa_paths=(
#   [bail]="best_overall_json/vanilla_nifa/sgc/bail.json"
#   [pokec_z]="best_overall_json/vanilla_nifa/sgc/pokec_z.json"
#   [pokec_n]="best_overall_json/vanilla_nifa/sgc/pokec_n.json"
#   [nba]="best_overall_json/vanilla_nifa/sgc/nba.json"
#   [german]="best_overall_json/vanilla_nifa/sgc/german.json"
# )

declare -A best_ea_paths=(
  [bail]="best_overall_json/ea_obj/sgc/bail.json"
  [pokec_z]="best_overall_json/ea_obj/sgc/pokec_z.json"
  [pokec_n]="best_overall_json/ea_obj/sgc/pokec_n.json"
  [nba]="best_overall_json/ea_obj/sgc/nba.json"
  [german]="best_overall_json/ea_obj/sgc/german.json"
)

# -----------------------------
# Common args
# -----------------------------
model="edge_adder"
attack="nifa"
nifa_mode="uncertainty"

epochs=500
start_seed=0
seed_num=10

allow_sgc_fallback=1

log_dir="logs/scaled_use_tuned_hp/tuned_ea/nifa_uncertainty/"

# Optional: your new pipeline flags (EdgeAdder only)
edge_pipeline="freeze_gnn_then_edge"   # or "joint"
edge_cand_source="emb"                # or "feat"
# pretrain_epochs=200
# edge_epochs=200

# -----------------------------
# Loops: sgc x 5 datasets
# -----------------------------
for encoder in sgc; do
  echo
  echo "================ ENCODER: ${encoder^^} ================"

  for dataset in pokec_z pokec_n nba bail german; do
    echo
    echo "============ ${dataset^^} ============="

    vanilla_tmpl="${best_vanilla_paths[$dataset]}"
    # nifa_tmpl="${best_nifa_paths[$dataset]}"
    ea_tmpl="${best_ea_paths[$dataset]}"

    best_vanilla_path="${vanilla_tmpl/\/sgc\//\/${encoder}\/}"
    # best_nifa_path="${nifa_tmpl/\/sgc\//\/${encoder}\/}"
    best_ea_path="${ea_tmpl/\/sgc\//\/${encoder}\/}"

    # -------- existence check + fallback to sgc --------
    if [[ ! -f "$best_vanilla_path" ]]; then
      echo "⚠️  Warning: vanilla best_overall.json not found at: $best_vanilla_path"
      if [[ "$allow_sgc_fallback" -eq 1 && -f "$vanilla_tmpl" ]]; then
        echo "    -> fallback to sgc tuned file: $vanilla_tmpl"
        best_vanilla_path="$vanilla_tmpl"
      else
        echo "    -> skip (no vanilla tuned file)."
        continue
      fi
    fi

    # if [[ ! -f "$best_nifa_path" ]]; then
    #   echo "⚠️  Warning: vanilla_nifa best_overall.json not found at: $best_nifa_path"
    #   if [[ "$allow_sgc_fallback" -eq 1 && -f "$nifa_tmpl" ]]; then
    #     echo "    -> fallback to sgc tuned file: $nifa_tmpl"
    #     best_nifa_path="$nifa_tmpl"
    #   else
    #     echo "    -> skip (no vanilla_nifa tuned file)."
    #     continue
    #   fi
    # fi

    if [[ ! -f "$best_ea_path" ]]; then
      echo "⚠️  Warning: ea_obj best_overall.json not found at: $best_ea_path"
      if [[ "$allow_sgc_fallback" -eq 1 && -f "$ea_tmpl" ]]; then
        echo "    -> fallback to sgc tuned file: $ea_tmpl"
        best_ea_path="$ea_tmpl"
      else
        echo "    -> skip (no ea_obj tuned file)."
        continue
      fi
    fi

    echo "Using tuned vanilla     : $best_vanilla_path"
    # echo "Using tuned vanilla_nifa: $best_nifa_path"
    echo "Using tuned ea_obj      : $best_ea_path"

    # -------- dataset-specific NIFA / fairness knobs --------
    if [[ "$dataset" == "pokec_n" ]]; then
      nifa_node=87
      nifa_edge=50
    elif [[ "$dataset" == "pokec_z" ]]; then
      nifa_node=102
      nifa_edge=50
    elif [[ "$dataset" == "bail" ]]; then
      nifa_node=25
      nifa_edge=50
    elif [[ "$dataset" == "nba" ]]; then
      nifa_node=4
      nifa_edge=15
    elif [[ "$dataset" == "german" ]]; then
      nifa_node=10
      nifa_edge=50
    fi

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
      --model $model \
      --encoder $encoder \
      --dataset $dataset \
      --start_seed $start_seed \
      --seed_num $seed_num \
      --epochs $epochs \
      --best_overall_path $best_vanilla_path $best_ea_path \
      --log_dir $log_dir \
      --attack $attack \
      --nifa_mode $nifa_mode --nifa_node $nifa_node --nifa_edge $nifa_edge \
      --nifa_alpha 0.01 --nifa_beta 4 --nifa_ratio 0.5 \
      --edge_pipeline $edge_pipeline \
      --edge_cand_source $edge_cand_source"
      # --pretrain_epochs $pretrain_epochs \
      # --edge_epochs $edge_epochs

    run_cmd "$cmd"
  done
done

echo
echo "✅ All runs finished."
