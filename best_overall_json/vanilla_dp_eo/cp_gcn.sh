#!/bin/bash

encoder="gcn"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/bail/gcn/vanilla/auc_f1_mean_minus_std/20251210-222019_vanilla-dp-eo-no-attack/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/pokec_z/gcn/vanilla/auc_f1_mean_minus_std/20251211-061423_vanilla-dp-eo-no-attack/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/pokec_n/gcn/vanilla/auc_f1_mean_minus_std/20251211-124559_vanilla-dp-eo-no-attack/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/nba/gcn/vanilla/auc_f1_mean_minus_std/20251211-210500_vanilla-dp-eo-no-attack/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_dp_eo/no_attack/german/gcn/vanilla/auc_f1_mean_minus_std/20251212-013140_vanilla-dp-eo-no-attack/best_overall.json"
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
