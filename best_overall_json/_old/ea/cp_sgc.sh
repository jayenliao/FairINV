encoder="sgc"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_ea/no_attack/bail/sgc/edge_adder/auc_f1_mean_minus_std/20251112-204225_ea-no-attack/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_ea/no_attack/pokec_z/sgc/edge_adder/auc_f1_mean_minus_std/20251113-092933_ea-no-attack/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_ea/no_attack/pokec_n/sgc/edge_adder/auc_f1_mean_minus_std/20251114-051924_ea-no-attack/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_ea/no_attack/nba/sgc/edge_adder/auc_f1_mean_minus_std/20251114-212617_ea-no-attack/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_ea/no_attack/german/sgc/edge_adder/auc_f1_mean_minus_std/20251115-013306_ea-no-attack/best_overall.json"
)

mkdir -p "best_overall_json/ea/$encoder"
for ds in "${datasets[@]}"; do
    echo "Copying best_overall.json for dataset: $ds"
    best_path="${best_paths[$ds]}"
    echo "From: $best_path"
    echo "To:   best_overall_json/ea/$encoder/${ds}.json"
    cp "$best_path" best_overall_json/ea/$encoder/"$ds".json
    echo
done
