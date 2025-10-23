datasets=(pokec_z pokec_n nba bail german)
backbones=(gcn gat gin sage sgc)
lambda_params=(lambda_dp lambda_eo)
baseline_model=vanilla

for dataset in "${datasets[@]}"; do
    for backbone in "${backbones[@]}"; do
        for lambda_param in "${lambda_params[@]}"; do
            echo "Generating plot for Dataset: $dataset, Backbone: $backbone, Lambda Param: $lambda_param"
            python visualize.py --csv_file ./ea_non-mm/ea_dp_or_eo_only_with_vanilla.csv \
                --dataset $dataset \
                --backbone $backbone \
                --lambda_param $lambda_param \
                --baseline_model $baseline_model \
                --save_dir ./figures/ \
                --save_fn ${dataset}_${backbone}_${lambda_param}.png
        done
    done
done
