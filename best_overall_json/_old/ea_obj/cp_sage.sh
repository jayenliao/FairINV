encoder="sage"
datasets=(bail pokec_z pokec_n nba german)
exp_name=ea_obj

declare -A best_paths=(
  [bail]="logs/tune_ea/no_attack_obj/bail/sage/edge_adder/auc_f1_balanced/20251118-103820_ea-no-attack/best_overall.json"
  [pokec_z]="logs/tune_ea/no_attack_obj/pokec_z/sage/edge_adder/auc_f1_balanced/20251118-143156_ea-no-attack/best_overall.json"
  [pokec_n]="logs/tune_ea/no_attack_obj/pokec_n/sage/edge_adder/auc_f1_balanced/20251118-194800_ea-no-attack/best_overall.json"
  [nba]="logs/tune_ea/no_attack_obj/nba/sage/edge_adder/auc_f1_balanced/20251119-034832_ea-no-attack/best_overall.json"
  [german]="logs/tune_ea/no_attack_obj/german/sage/edge_adder/auc_f1_balanced/20251119-063608_ea-no-attack/best_overall.json"
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
