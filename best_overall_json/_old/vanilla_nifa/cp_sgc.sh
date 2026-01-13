encoder="sgc"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/bail/sgc/vanilla/attack_balanced/20251109-162254_nifa/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_z/sgc/vanilla/attack_balanced/20251111-131842_nifa/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_n/sgc/vanilla/attack_balanced/20251110-055638_nifa/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/nba/sgc/vanilla/attack_balanced/20251111-233408_nifa/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/german/sgc/vanilla/attack_balanced/20251112-033126_nifa/best_overall.json"
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
