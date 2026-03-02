GPU_ID=7 EPOCHS=200 SEED_NUM=10 ADV_K=5 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gat" \
    ADVTRAIN_ATTACK=edge_weight ADV_EDGE_POLICY=same_largest \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=7 EPOCHS=200 SEED_NUM=10 ADV_K=5 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gcn" \
    ADVTRAIN_ATTACK=edge_weight ADV_EDGE_POLICY=same_largest \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=6 EPOCHS=200 SEED_NUM=10 ADV_K=5 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gin" \
    ADVTRAIN_ATTACK=edge_weight ADV_EDGE_POLICY=same_largest \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=6 EPOCHS=200 SEED_NUM=10 ADV_K=5 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight ADV_EDGE_POLICY=same_largest \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=6 EPOCHS=200 SEED_NUM=10 ADV_K=5 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sgc" \
    ADVTRAIN_ATTACK=edge_weight ADV_EDGE_POLICY=same_largest \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh
