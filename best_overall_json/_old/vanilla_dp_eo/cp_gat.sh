#!/bin/bash

encoder="gat"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/bail/gat/vanilla/auc_f1_mean_minus_std/20251210-221915_vanilla-dp-eo-no-attack/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/pokec_z/gat/vanilla/auc_f1_mean_minus_std/20251211-051457_vanilla-dp-eo-no-attack/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/pokec_n/gat/vanilla/auc_f1_mean_minus_std/20251211-124528_vanilla-dp-eo-no-attack/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/nba/gat/vanilla/auc_f1_mean_minus_std/20251211-215758_vanilla-dp-eo-no-attack/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/german/gat/vanilla/auc_f1_mean_minus_std/20251211-235142_vanilla-dp-eo-no-attack/best_overall.json"
)

mkdir -p vanilla_dp_eo/"$encoder"
for ds in "${datasets[@]}"; do
    echo "Copying best_overall.json for dataset: $ds"
    best_path="${best_paths[$ds]}"
    echo "From: $best_path"
    echo "To:   vanilla_dp_eo/$encoder/${ds}.json"
    cp "$best_path" vanilla_dp_eo/$encoder/"$ds".json
    echo
done
