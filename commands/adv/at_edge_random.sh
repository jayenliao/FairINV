GPU_ID=6 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=5 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="gat" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=5 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=5 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gcn" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=4 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=5 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gin" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=3 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=5 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=5 \
    DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sgc" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh
