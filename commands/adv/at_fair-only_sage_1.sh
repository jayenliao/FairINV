GPU_ID=3 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="same_largest" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=4 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=3 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="same_smallest" \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=4 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

