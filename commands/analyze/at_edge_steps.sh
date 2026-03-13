log_root="logs/advtrain_edge_fairness_only"
exp_folders=(
    edge_weight_mode-mix_k-5_steps-2_dp-2.0_eo-2.0_policy-global_random
    edge_weight_mode-mix_k-5_steps-8_dp-2.0_eo-2.0_policy-global_random
    edge_weight_mode-mix_k-5_steps-12_dp-2.0_eo-2.0_policy-global_random
    edge_weight_mode-mix_k-5_steps-16_dp-2.0_eo-2.0_policy-global_random
)

for exp_folder in "${exp_folders[@]}"; do
    echo "Processing ${exp_folder}..."
    python analysis/analysis.py "${log_root}/${exp_folder}" \
        --split_name test \
        --output auto
done

for exp_folder in "${exp_folders[@]}"; do
    echo "Processing ${exp_folder}..."
    python analysis/analysis.py "${log_root}/${exp_folder}" \
        --split_name test_clean \
        --output auto
done
