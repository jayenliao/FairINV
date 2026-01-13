encoder="sgc"
datasets=(bail pokec_z pokec_n nba german)

declare -A best_paths=(
  [bail]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/bail/sgc/vanilla/auc_f1/20251007-190642_auto75/best_overall.json"
  [pokec_z]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_z/sgc/vanilla/auc_f1/20251007-234448_auto75/best_overall.json"
  [pokec_n]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/pokec_n/sgc/vanilla/auc_f1/20251007-221300_auto75/best_overall.json"
  [nba]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/nba/sgc/vanilla/auc_f1/20251007-174119_auto75/best_overall.json"
  [german]="/tmp2/jayliao/gnn_fairness/FairINV/logs_piplup/optuna/german/sgc/vanilla/auc_f1/20251007-173256_auto75/best_overall.json"
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
