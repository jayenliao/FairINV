encoder="gin"
datasets=(bail pokec_z pokec_n nba german)
exp_name=ea_obj

declare -A best_paths=(
  [bail]="logs/tune_ea/no_attack_obj/bail/gin/edge_adder/auc_f1_balanced/20251118-12055_ea-no-attack/best_overall.json"
  [pokec_z]="logs/tune_ea/no_attack_obj/pokec_z/gin/edge_adder/auc_f1_balanced/20251118-140057_ea-no-attack/best_overall.json"
  [pokec_n]="logs/tune_ea/no_attack_obj/pokec_n/gin/edge_adder/auc_f1_balanced/20251118-193606_ea-no-attack/best_overall.json"
  [nba]="logs/tune_ea/no_attack_obj/nba/gin/edge_adder/auc_f1_balanced/20251119-015636_ea-no-attack/best_overall.json"
  [german]="logs/tune_ea/no_attack_obj/german/gin/edge_adder/auc_f1_balanced/20251119-045920_ea-no-attack/best_overall.json"
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
