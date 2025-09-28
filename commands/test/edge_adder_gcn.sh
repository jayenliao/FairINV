echo 'Running EdgeAdder GCN for 5 runs with different random seeds...'

echo
echo '============German============='
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model edge_adder --encoder gcn --dataset german \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 3 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.1 \
    --log_dir logs/test/edge_adder

echo
echo '============Bail============='
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model edge_adder --encoder gcn --dataset bail \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 3 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.1 \
    --log_dir logs/test/edge_adder

echo
echo '============Pokec_z============='
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model edge_adder --encoder gcn --dataset pokec_z \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 3 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.01 \
    --log_dir logs/test/edge_adder

echo
echo '============Pokec_n============='
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model edge_adder --encoder gcn --dataset pokec_n \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 3 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 1 --lr_sp 0.5 \
    --log_dir logs/test/edge_adder

echo
echo '============nba============='
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model edge_adder --encoder gcn --dataset nba \
    --edge_k 2 --lambda_dp 0.1 --lambda_edge_l1 1e-4 \
    --start_seed 0 --seed_num 5 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 1 --lr_sp 0.1 \
    --log_dir logs/test/edge_adder
