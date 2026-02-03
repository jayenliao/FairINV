import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    parser.add_argument('--model', choices=['fairinv', 'vanilla', 'edge_adder', 'edge_minmax'], default='vanilla')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation.')
    parser.add_argument('--start_seed', type=int, default=42, help='Random seed start.')
    parser.add_argument('--seed_num', type=int, default=10, help='The number of random seed.')
    parser.add_argument('--num_threads', type=int, default=1,
                        help="Number of CPU threads to use for BLAS/DGL/PyTorch ops.")
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hid_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['nba', 'bail', 'pokec_z', 'pokec_n', 'german'])
    parser.add_argument("--layer_num", type=int, default=2, help="number of hidden layers")
    parser.add_argument('--encoder', type=str, default='gcn', choices=['gcn','gat','gin','sage','sgc'])
    parser.add_argument('--aggr', type=str, default='add',
                        choices=['add', 'mean', 'max', 'min', 'sum', 'std', 'var', 'median'],
                        help="aggregation function")
    parser.add_argument('--weight_path', type=str, default='./weights/model_weight.pt')
    parser.add_argument('--save_results', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='hyperpapameter to balance the downstream task and invariance learning loss.')
    parser.add_argument('--lr_sp', type=float, default=0.5, help='the learning rate of the sensitive partition.')
    parser.add_argument('--env_num', type=int, default=2,
                        help='the number of the sensitive attribute, also known as environment number.')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--log_interval', type=int, default=20, help='Interval for logging.')
    parser.add_argument('--partition_times', type=int, default=3,
                        help='the number for partitioning the sensitive attribute group.')
    parser.add_argument('--use_neg_metrics', action='store_true',
                        help='Whether to use negative sampling for computing fairness metrics.')
    parser.add_argument('--lambda_dp', type=float, default=0.0, help='Weight for soft demographic parity loss.')
    parser.add_argument('--lambda_eo', type=float, default=0.0, help='Weight for soft equal opportunity loss.')

    # edge adder specific
    parser.add_argument('--edge_k', type=int, default=2, help='#candidate pairs per node.')
    parser.add_argument('--eo_mode', type=str, choices=['tpr','fpr','both'], default='tpr',
                        help='Mode for equal opportunity loss.')
    parser.add_argument('--lambda_edge_l1', type=float, default=1e-4, help='L1 sparsity on learnable edges.')
    # edge adder pipeline variants
    parser.add_argument('--edge_pipeline', type=str, choices=['joint','freeze_gnn_then_edge'], default='freeze_gnn_then_edge',
                        help="EdgeAdder training pipeline: 'joint' or 'freeze_gnn_then_edge'.")
    parser.add_argument('--pretrain_epochs', type=int, default=0,
                        help='Stage-1 epochs (GNN pretrain). 0 => use --epochs.')
    parser.add_argument('--edge_epochs', type=int, default=0,
                        help='Stage-3 epochs (EdgeAdder training). 0 => use --epochs.')

    # alternating (iterative) training for EdgeAdder baseline under freeze_gnn_then_edge
    # Each round alternates:
    #   (1) freeze GNN/clf, update edge weights for --alt_edge_epochs
    #   (2) freeze edges, update GNN/clf on blended graph for --alt_gnn_epochs
    # Enable by setting --alt_rounds > 0.
    parser.add_argument('--alt_rounds', type=int, default=0,
                        help='Number of alternation rounds (0 disables alternating training).')
    parser.add_argument('--alt_edge_epochs', type=int, default=0,
                        help='Edge-weight update epochs per round (0 => use --edge_epochs if set, else 20).')
    parser.add_argument('--alt_gnn_epochs', type=int, default=0,
                        help='GNN update epochs per round (0 => use --pretrain_epochs if set, else 20).')
    parser.add_argument('--alt_gnn_lr', type=float, default=None,
                        help='Learning rate for the GNN update step in alternating training (default: --lr).')
    parser.add_argument('--pretrain_lambda_dp', type=float, default=None,
                        help='Override lambda_dp during stage-1 pretraining. Default=None uses --lambda_dp.')
    parser.add_argument('--pretrain_lambda_eo', type=float, default=None,
                        help='Override lambda_eo during stage-1 pretraining. Default=None uses --lambda_eo.')
    parser.add_argument('--edge_cand_source', type=str, choices=['feat','emb'], default=None,
                        help="Candidate feature: 'feat' (raw) or 'emb' (pretrained). Default: joint->feat, freeze->emb.")
    parser.add_argument('--adv_reduce_exclude_l1', action='store_true',
                        help='When picking the worst policy, exclude L1 from the per-policy objective.')
    parser.add_argument('--scale_lambda', type=int, default=2,
                        help='Scale up lambda_dp and lambda_eo by this factor to match the magnitude of BCE loss.')

    # edge minmax specific
    parser.add_argument("--policy_names", nargs='+',
                        default=["same_largest", "cross_smallest", "same_smallest", "cross_random", "same_random"],
                        help="Edge selection policies to use.")
    parser.add_argument('--max_reduce', type=str, choices=['max','logsumexp'], default='max')
    parser.add_argument('--lse_tau', type=float, default=0.5)

    # load tuned HPs from Optuna output
    parser.add_argument('--best_overall_path', type=str, default=[], nargs='+',
                        help='Path to an Optuna best_overall.json; if set, override lr/weight_decay/.. from it.')

    # --- Attack toggle ---
    parser.add_argument('--attack', choices=['none', 'nifa'], default='none',
                        help="Optional pre-training attack pipeline. 'nifa' = node+edge injection (NIFA).")
    parser.add_argument('--attack_when', choices=['train','eval','both'], default='train',
                        help="When to apply attack: 'train'=poisoning (pre-training), 'eval'=evasion (eval-time only), 'both'=apply at both train and eval.")

    # --- NIFA hyperparameters (namespaced to avoid conflicts) ---
    parser.add_argument('--nifa_T', type=int, default=20)
    parser.add_argument('--nifa_theta', type=float, default=0.5)
    parser.add_argument('--nifa_node', type=int, default=102)     # injected nodes
    parser.add_argument('--nifa_edge', type=int, default=50)      # degree budget (even number)
    parser.add_argument('--nifa_alpha', type=float, default=1.0)
    parser.add_argument('--nifa_beta', type=float, default=1.0)
    parser.add_argument('--nifa_ratio', type=float, default=0.5)  # top-ratio uncertain nodes to target
    parser.add_argument('--nifa_mode', choices=['uncertainty','degree'], default='uncertainty')
    parser.add_argument('--nifa_epochs', type=int, default=1000)
    parser.add_argument('--nifa_lr', type=float, default=0.001)
    parser.add_argument('--nifa_loops', type=int, default=50)
    parser.add_argument('--nifa_gamma', type=float, default=1.0,
                    help="Weight on task utility (CE) inside NIFA objective. 0 => utility-agnostic attack.")
    parser.add_argument('--nifa_keep_markers', action='store_true',
                        help="Keep injected node markers (label=-1, sens=-1). Default: sanitize markers to avoid leaking which nodes are injected.")

    # --- Adversarial training defense ---
    parser.add_argument('--advtrain', action='store_true',
                        help='Enable adversarial training defense (generate attacked graphs during training).')
    parser.add_argument('--advtrain_attack', choices=['none','nifa'], default='nifa',
                        help='Adversary used for generating training-time attacked graphs.')
    parser.add_argument('--advtrain_mode', choices=['mix','robust'], default='mix',
                        help="mix: L_clean + lambda_adv * mean(L_adv). robust: reduce over {clean, adv_i}.")
    parser.add_argument('--advtrain_k', type=int, default=1,
                        help='Number of attacked graph variants per epoch.')
    parser.add_argument('--advtrain_gen', choices=['precompute','on_the_fly'], default='precompute',
                        help='How to generate attacked graphs.')
    parser.add_argument('--advtrain_refresh', type=int, default=0,
                        help='When advtrain_gen=on_the_fly, regenerate every N epochs (0 disables).')
    parser.add_argument('--advtrain_cache_device', action='store_true',
                        help='Cache generated attacked variants on GPU (faster, more memory).')

    parser.add_argument('--advtrain_mix_lambda', type=float, default=1.0,
                        help='lambda_adv used by advtrain_mode=mix.')
    parser.add_argument('--advtrain_include_clean', action='store_true',
                        help='Include clean graph in robust reduction (advtrain_mode=robust).')
    parser.add_argument('--advtrain_reduce', choices=['mean','max','logsumexp'], default='max',
                        help='Reduction across attacked variants (and clean if included) for robust training.')
    parser.add_argument('--advtrain_tau', type=float, default=0.5,
                        help='Temperature for logsumexp reduction.')
    parser.add_argument('--advtrain_seed_stride', type=int, default=1000,
                        help='Seed stride for generating different attack variants.')

    # Optional per-variant NIFA budgets (length 1 or length K; broadcasts if length=1)
    parser.add_argument('--advtrain_nifa_node', nargs='+', type=int, default=None,
                        help='Per-variant injected node counts.')
    parser.add_argument('--advtrain_nifa_edge', nargs='+', type=int, default=None,
                        help='Per-variant injected edge budgets.')
    parser.add_argument('--advtrain_nifa_ratio', nargs='+', type=float, default=None,
                        help='Per-variant target ratio.')
    parser.add_argument('--advtrain_nifa_gamma', nargs='+', type=float, default=None,
                        help='Per-variant NIFA gamma (utility weight).')

    # EdgeAdder option: whether candidates may involve injected nodes
    parser.add_argument('--edge_include_injected', action='store_true',
                        help='If set, allow edge-candidate construction to include NIFA-injected nodes. Default: ignore injected nodes.')

    return parser
