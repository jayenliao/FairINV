python train.py \
  --model vanilla \
  --advtrain --advtrain_attack edge_weight \
  --advtrain_mode robust --advtrain_k 3 \
  --advtrain_edge_policy cross_smallest \
  --advtrain_edge_steps 5 --advtrain_edge_step_size 0.1 \
  --advtrain_edge_grad sign \
  --advtrain_edge_w_max 1.0 \
  --advtrain_edge_budget -1
