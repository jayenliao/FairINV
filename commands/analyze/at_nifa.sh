exp_folders=(
    "compare_expB_k-2_lambda_DP-1.0_lambda_EO-1.0"
    # "compare_expB_k-4_lambda_DP-1.0_lambda_EO-1.0"
    "compare_expB_k-6_lambda_DP-1.0_lambda_EO-1.0"
    "compare_expB_k-8_lambda_DP-1.0_lambda_EO-1.0"
    "compare_expB_k-10_lambda_DP-1.0_lambda_EO-1.0"
    "compare_expB_k-12_lambda_DP-1.0_lambda_EO-1.0"
    # "compare_expB_k-14_lambda_DP-1.0_lambda_EO-1.0"
    "compare_expB_k-16_lambda_DP-1.0_lambda_EO-1.0"
)

for exp_folder in "${exp_folders[@]}"; do
    echo "Processing ${exp_folder}..."
    python analysis/analysis.py "logs/${exp_folder}" \
        --split_name test \
        --output auto
done

for exp_folder in "${exp_folders[@]}"; do
    echo "Processing ${exp_folder}..."
    python analysis/analysis.py "logs/${exp_folder}" \
        --split_name test_clean \
        --output auto
done
