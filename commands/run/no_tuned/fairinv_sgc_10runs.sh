echo 'Running FairINV sgc for 5 runs with different random seeds...'

echo
echo '============German============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder sgc --dataset german \
    --start_seed 42 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.1

echo
echo '============Bail============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder sgc --dataset bail \
    --start_seed 42 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.1

echo
echo '============Pokec_z============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder sgc --dataset pokec_z \
    --start_seed 42 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.01

echo
echo '============Pokec_n============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder sgc --dataset pokec_n \
    --start_seed 42 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 1 --lr_sp 0.5

echo
echo '============nba============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder sgc --dataset nba \
    --start_seed 42 --seed_num 10 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 1 --lr_sp 0.1
