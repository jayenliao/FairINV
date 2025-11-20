encoder="sgc"
datasets=(bail pokec_z pokec_n nba german)
exp_name=ea_obj

declare -A best_paths=(
  [bail]="logs/tune_ea/no_attack_obj/bail/sgc/edge_adder/auc_f1_balanced/20251118-103823_ea-no-attack/best_overall.json"
  [pokec_z]="logs/tune_ea/no_attack_obj/pokec_z/sgc/edge_adder/auc_f1_balanced/20251118-150449_ea-no-attack/best_overall.json"
  [pokec_n]="logs/tune_ea/no_attack_obj/pokec_n/sgc/edge_adder/auc_f1_balanced/20251118-225714_ea-no-attack/best_overall.json"
  [nba]="logs/tune_ea/no_attack_obj/nba/sgc/edge_adder/auc_f1_balanced/20251119-061440_ea-no-attack/best_overall.json"
  [german]="logs/tune_ea/no_attack_obj/german/sgc/edge_adder/auc_f1_balanced/20251119-084354_ea-no-attack/best_overall.json"
)

mkdir -p "best_overall_json/$exp_name/$encoder"
for ds in "${datasets[@]}"; do
    echo "Copying best_overall.json for dataset: $ds"
    best_path="${best_paths[$ds]}"
    echo "From: $best_path"
    echo "To:   best_overall_json/$exp_name/$encoder/${ds}.json"
    cp "$best_path" best_overall_json/$exp_name/$encoder/"$ds".json
    echo
done
