CUDA_VISIBLE_DEVICES=0

echo 'Running EdgeMinMax with Min-Max game gat for 5 runs with different random seeds...'

echo
echo '============German============='
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --model edge_minmax --encoder gat --dataset german \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 \
    --max_reduce logsumexp --lse_tau 0.5 \
    --policy_names same_largest cross_smallest same_smallest cross_random same_random \
    --log_dir logs/runs/edge_minmax

echo
echo '============Bail============='
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --model edge_minmax --encoder gat --dataset bail \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 \
    --max_reduce logsumexp --lse_tau 0.5 \
    --policy_names same_largest cross_smallest same_smallest cross_random same_random \
    --log_dir logs/runs/edge_minmax

echo
echo '============Pokec_z============='
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --model edge_minmax --encoder gat --dataset pokec_z \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 \
    --max_reduce logsumexp --lse_tau 0.5 \
    --policy_names same_largest cross_smallest same_smallest cross_random same_random \
    --log_dir logs/runs/edge_minmax

echo
echo '============Pokec_n============='
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --model edge_minmax --encoder gat --dataset pokec_n \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 \
    --max_reduce logsumexp --lse_tau 0.5 \
    --policy_names same_largest cross_smallest same_smallest cross_random same_random \
    --log_dir logs/runs/edge_minmax

echo
echo '============nba============='
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train.py \
    --model edge_minmax --encoder gat --dataset nba \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 5 \
    --hid_dim 16 --lr 1e-2 --epochs 1000  \
    --max_reduce logsumexp --lse_tau 0.5 \
    --policy_names same_largest cross_smallest same_smallest cross_random same_random \
    --log_dir logs/runs/edge_minmax
