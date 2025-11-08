encoder="gat"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/bail/gat/vanilla/auc_f1/20251006-151405_auto75/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_z/gat/vanilla/auc_f1/20251006-210344_auto75/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_n/gat/vanilla/auc_f1/20251006-164201_auto75/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/nba/gat/vanilla/auc_f1/20251006-165848_auto75/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/german/gat/vanilla/auc_f1/20251006-101506_auto75/best_overall.json"
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
