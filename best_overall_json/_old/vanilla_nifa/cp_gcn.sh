encoder="gcn"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/bail/gcn/vanilla/attack_balanced/20251109-162033_nifa/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_z/gcn/vanilla/attack_balanced/20251110-151328_nifa/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_n/gcn/vanilla/attack_balanced/20251110-000134_nifa/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/nba/gcn/vanilla/attack_balanced/20251111-000302_nifa/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/german/gcn/vanilla/attack_balanced/20251111-042921_nifa/best_overall.json"
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
