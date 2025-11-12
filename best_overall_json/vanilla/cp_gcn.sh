encoder="gcn"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/bail/gcn/vanilla/auc_f1/20251006-003953_auto75/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_z/gcn/vanilla/auc_f1/20251006-004128_auto75/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_n/gcn/vanilla/auc_f1/20251006-004151_auto75/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/nba/gcn/vanilla/auc_f1/20251006-004228_auto75/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/german/gcn/vanilla/auc_f1/20251006-003808_auto75/best_overall.json"
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
