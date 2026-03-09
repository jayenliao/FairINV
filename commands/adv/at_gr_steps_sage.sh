GPU_ID=2 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    OMP_NUM_THREADS=12 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=2 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    OMP_NUM_THREADS=12 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=8 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    OMP_NUM_THREADS=12 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=12 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 ADV_K=5 ADV_EDGE_POLICY="global_random" \
    OMP_NUM_THREADS=12 \
    LAMBDA_DP=2.0 LAMBDA_EO=2.0 ADV_EDGE_STEPS=16 \
    DATASETS="bail pokec_z pokec_n" ENCODERS="sage" \
    ADVTRAIN_ATTACK=edge_weight \
    bash commands/adv/advtrain_edge_clean_and_attacked.sh
