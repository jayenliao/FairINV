echo 'Running FairINV GIN for 10 runs with different random seeds...'

echo
echo '============German============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder gin --dataset german \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --seed_num 10 --alpha 10 --lr_sp 0.1

echo
echo '============Bail============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder gin --dataset bail \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --seed_num 10 --alpha 10 --lr_sp 0.1

echo
echo '============Pokec_z============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder gin --dataset pokec_z \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --seed_num 10 --alpha 10 --lr_sp 0.01

echo
echo '============Pokec_n============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder gin --dataset pokec_n \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --seed_num 10 --alpha 1 --lr_sp 0.5

echo
echo '============nba============='
CUDA_VISIBLE_DEVICES=2 python train.py \
    --model fairinv --encoder gin --dataset nba \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --seed_num 10 --alpha 1 --lr_sp 0.1
