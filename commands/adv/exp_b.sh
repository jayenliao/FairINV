GPU_ID=3 EPOCHS=200 SEED_NUM=10 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gat" \
bash commands/adv/compare_edgeadder_vs_advtrain_expB.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gcn" \
bash commands/adv/compare_edgeadder_vs_advtrain_expB.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="gin" \
bash commands/adv/compare_edgeadder_vs_advtrain_expB.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sage" \
bash commands/adv/compare_edgeadder_vs_advtrain_expB.sh

GPU_ID=2 EPOCHS=200 SEED_NUM=10 DATASETS="bail pokec_z pokec_n german nba" ENCODERS="sgc" \
bash commands/adv/compare_edgeadder_vs_advtrain_expB.sh
