GPU_ID=4 EPOCHS=200 SEED_NUM=10 ADV_K=14 LAMBDA_DP=1.0 LAMBDA_EO=1.0 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gat" \
bash commands/adv/only_advtrain_expB.sh

GPU_ID=6 EPOCHS=200 SEED_NUM=10 ADV_K=14 LAMBDA_DP=1.0 LAMBDA_EO=1.0 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gcn" \
bash commands/adv/only_advtrain_expB.sh

GPU_ID=6 EPOCHS=200 SEED_NUM=10 ADV_K=14 LAMBDA_DP=1.0 LAMBDA_EO=1.0 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gin" \
bash commands/adv/only_advtrain_expB.sh

GPU_ID=7 EPOCHS=200 SEED_NUM=10 ADV_K=14 LAMBDA_DP=1.0 LAMBDA_EO=1.0 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sage" \
bash commands/adv/only_advtrain_expB.sh

GPU_ID=4 EPOCHS=200 SEED_NUM=10 ADV_K=14 LAMBDA_DP=1.0 LAMBDA_EO=1.0 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sgc" \
bash commands/adv/only_advtrain_expB.sh
