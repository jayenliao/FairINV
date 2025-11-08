CUDA_VISIBLE_DEVICES=1
dataset="pokec_z"
encoder="gcn"
model="edge_adder"
attack="nifa"
epochs=1000
start_seed=0
seed_num=10
lambda_dp=1.0
lambda_eo=0.0
best_path=logs_piplup/optuna/pokec_z/gcn/vanilla/auc_f1/20251006-004128_auto75/best_overall.json
log_dir="logs/test/tuned_vanilla/nifa"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --model "$model" \
    --encoder "$encoder" \
    --dataset "$dataset" \
    --start_seed "$start_seed" \
    --seed_num "$seed_num" \
    --epochs "$epochs" \
    --best_overall_path "$best_path" \
    --lambda_dp "$lambda_dp" \
    --lambda_eo "$lambda_eo" \
    --log_dir "$log_dir" \
    --attack "$attack" \
    --nifa_mode 'degree' --nifa_node 102 --nifa_edge 50 \
    --nifa_alpha 0.01 --nifa_beta 4 --nifa_ratio 0.5
