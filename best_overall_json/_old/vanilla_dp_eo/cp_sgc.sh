#!/bin/bash

encoder="sgc"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/bail/sgc/vanilla/auc_f1_mean_minus_std/20251210-222453_vanilla-dp-eo-no-attack/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/pokec_z/sgc/vanilla/auc_f1_mean_minus_std/20251211-062105_vanilla-dp-eo-no-attack/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/pokec_n/sgc/vanilla/auc_f1_mean_minus_std/20251211-124730_vanilla-dp-eo-no-attack/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/nba/sgc/vanilla/auc_f1_mean_minus_std/20251211-194413_vanilla-dp-eo-no-attack/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/german/sgc/vanilla/auc_f1_mean_minus_std/20251212-001223_vanilla-dp-eo-no-attack/best_overall.json"
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
