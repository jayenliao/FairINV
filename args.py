import argparse
from html import parser
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Whether to run in debug mode.')
    parser.add_argument('--model', choices=['fairinv', 'vanilla', 'edge_adder', 'edge_minmax'], default='vanilla')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
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

    # edge adder specific
    parser.add_argument('--edge_k', type=int, default=2, help='#candidate pairs per node.')
    parser.add_argument('--lambda_dp', type=float, default=0.1, help='Weight for soft demographic parity loss.')
    parser.add_argument('--lambda_eo', type=float, default=0.0, help='Weight for soft equal opportunity loss.')
    parser.add_argument('--lambda_edge_l1', type=float, default=1e-4, help='L1 sparsity on learnable edges.')
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
    parser.add_argument('--best_overall_path', type=str, default='',
                        help='Path to an Optuna best_overall.json; if set, override lr/weight_decay/.. from it.')

    # --- Attack toggle ---
    parser.add_argument('--attack', choices=['none', 'nifa'], default='none',
                        help="Optional pre-training attack pipeline. 'nifa' = node+edge injection (NIFA).")

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

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args
