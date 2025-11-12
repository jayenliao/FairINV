encoder="gin"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/bail/gin/vanilla/attack_balanced/20251109-162227_nifa/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_z/gin/vanilla/attack_balanced/20251111-131726_nifa/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/pokec_n/gin/vanilla/attack_balanced/20251109-235909_nifa/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/nba/gin/vanilla/attack_balanced/20251111-224058_nifa/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs/tune_vanilla_nifa/german/gin/vanilla/attack_balanced/20251112-022908_nifa/best_overall.json"
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
