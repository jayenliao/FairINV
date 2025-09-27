echo 'Running Vanilla sage for 5 runs with different random seeds...'

echo
echo '============German============='
CUDA_VISIBLE_DEVICES=2 python train_fairinv.py \
    --model vanilla --encoder sage --dataset german \
    --start_seed 0 --seed_num 5 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.1

echo ''
echo '============Bail============='
CUDA_VISIBLE_DEVICES=2 python train_fairinv.py \
    --model vanilla --encoder sage --dataset bail \
    --start_seed 0 --seed_num 5 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.1

echo ''
echo '============Pokec_z============='
CUDA_VISIBLE_DEVICES=2 python train_fairinv.py \
    --model vanilla --encoder sage --dataset pokec_z \
    --start_seed 0 --seed_num 5 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 10 --lr_sp 0.01

echo
echo '============Pokec_n============='
CUDA_VISIBLE_DEVICES=2 python train_fairinv.py \
    --model vanilla --encoder sage --dataset pokec_n \
    --start_seed 0 --seed_num 5 \
    --hid_dim 16 --lr 1e-2 --epochs 1000 --alpha 1 --lr_sp 0.5
