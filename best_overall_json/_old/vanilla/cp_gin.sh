encoder="gin"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/bail/gin/vanilla/auc_f1/20251007-103133_auto75/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_z/gin/vanilla/auc_f1/20251007-163832_auto75/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_n/gin/vanilla/auc_f1/20251007-135923_auto75/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/nba/gin/vanilla/auc_f1/20251007-100231_auto75/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/german/gin/vanilla/auc_f1/20251007-041741_auto75/best_overall.json"
)

mkdir -p vanilla/"$encoder"
for ds in "${datasets[@]}"; do
    echo "Copying best_overall.json for dataset: $ds"
    best_path="${best_paths[$ds]}"
    echo "From: $best_path"
    echo "To:   vanilla/$encoder/${ds}.json"
    cp "$best_path" vanilla/$encoder/"$ds".json
    echo
done
