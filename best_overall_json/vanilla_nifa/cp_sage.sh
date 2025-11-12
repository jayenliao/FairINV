encoder="sage"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/bail/sage/vanilla/attack_balanced/20251109-162056_nifa/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_z/sage/vanilla/attack_balanced/20251110-112714_nifa/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_n/sage/vanilla/attack_balanced/20251109-215800_nifa/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/nba/sage/vanilla/attack_balanced/20251110-200550_nifa/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/german/sage/vanilla/attack_balanced/20251111-004634_nifa/best_overall.json"
)

mkdir -p "$encoder"
for ds in "${datasets[@]}"; do
    echo "Copying best_overall.json for dataset: $ds"
    best_path="${best_paths[$ds]}"
    echo "From: $best_path"
    echo "To:   $encoder/${ds}.json"
    cp "$best_path" $encoder/"$ds".json
    echo
done
