encoder="gat"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/bail/gat/vanilla/attack_balanced/20251109-162001_nifa/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_z/gat/vanilla/attack_balanced/20251110-065315_nifa/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_n/gat/vanilla/attack_balanced/20251109-232512_nifa/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/nba/gat/vanilla/attack_balanced/20251110-152911_nifa/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/german/gat/vanilla/attack_balanced/20251110-185527_nifa/best_overall.json"
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
