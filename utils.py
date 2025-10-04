import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os, json, random
from torch.nn.modules.loss import _Loss
from scipy.spatial import distance_matrix
from torch.autograd import grad
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def set_seed(seed, use_cuda:bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)
    return idx_map


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2 * (features - min_values).div(max_values - min_values) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


class Results:
    def __init__(self, seed_num, model_num, args):
        super(Results, self).__init__()

        self.seed_num = seed_num
        self.model_num = model_num
        self.dataset = args.dataset
        self.model = args.model
        self.encoder = args.encoder
        self.auc, self.f1, self.acc, self.parity, self.equality = np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num)), \
                                                                  np.zeros(shape=(self.seed_num, self.model_num))

    def report_results(self):
        for i in range(self.model_num):
            print(f"============ {self.dataset} + {self.model} ({self.encoder}) ============")
            print(f"AUCROC:   {np.around(np.mean(self.auc[:, i]) * 100, 2)} ± {np.around(np.std(self.auc[:, i]) * 100, 2)}")
            print(f'F1-score: {np.around(np.mean(self.f1[:, i]) * 100, 2)} ± {np.around(np.std(self.f1[:, i]) * 100, 2)}')
            print(f'ACC:      {np.around(np.mean(self.acc[:, i]) * 100, 2)} ± {np.around(np.std(self.acc[:, i]) * 100, 2)}')
            print(f'Parity:   {np.around(np.mean(self.parity[:, i]) * 100, 2)} ± {np.around(np.std(self.parity[:, i]) * 100, 2)}')
            print(f'Equality: {np.around(np.mean(self.equality[:, i]) * 100, 2)} ± {np.around(np.std(self.equality[:, i]) * 100, 2)}')
            print("=================END=================")

    def save_results(self, args):
        for i in range(self.model_num):
            if hasattr(args, 'pbar'):
                del args.pbar
            args_dict = {k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(args).items()}
            save_path = os.path.join(args.log_dir, f"results_among_{args.seed_num}_seeds.json")

            ret_dict = {
                "AUC_mean": np.mean(self.auc[:, i]),
                "AUC_std":  np.std(self.auc[:, i]),
                "F1_mean":  np.mean(self.f1[:, i]),
                "F1_std":   np.std(self.f1[:, i]),
                "ACC_mean": np.mean(self.acc[:, i]),
                "ACC_std":  np.std(self.acc[:, i]),
                'DP_mean':  np.mean(self.parity[:, i]),
                'DP_std':   np.std(self.parity[:, i]),
                'EO_mean':  np.mean(self.equality[:, i]),
                'EO_std':   np.std(self.equality[:, i])
            }
            ret_dict['name'] = f"{args.model}_" + args.dataset + f"_{args.encoder}" + f"_alpha:{args.alpha}" + f"_lr_sp:{args.lr_sp}" + f"_env_num:{args.env_num}" + f"_lr:{args.lr}"

            output_dict = {"args": args_dict, "results": ret_dict}
            with open(save_path, 'w') as file:
                json.dump(output_dict, file, indent=4)

            # ret_dict_pretty = {
            #     "AUC": f"{np.around(ret_dict['AUC_mean'] * 100, 2)} ± {np.around(ret_dict['AUC_std'] * 100, 2)}",
            #     "F1": f"{np.around(ret_dict['F1_mean'] * 100, 2)} ± {np.around(ret_dict['F1_std'] * 100, 2)}",
            #     "ACC": f"{np.around(ret_dict['ACC_mean'] * 100, 2)} ± {np.around(ret_dict['ACC_std'] * 100, 2)}",
            #     'SP': f'{np.around(ret_dict["SP_mean"] * 100, 2)} ± {np.around(ret_dict["SP_std"] * 100, 2)}',
            #     'EO': f'{np.around(ret_dict["EO_mean"] * 100, 2)} ± {np.around(ret_dict["EO_std"] * 100, 2)}'
            # }
            # for k, v in ret_dict_pretty.items():
            #     print(f"{k:<5}: {v}")

            # with open('results.json', 'a') as file:
            #     json.dump(ret_dict, file, indent=4, ensure_ascii=False)
            #     file.write('\n')

def get_metrics(Y, logit, pred, idx, data):
    auc = roc_auc_score(Y[idx].cpu(), logit[idx].cpu())
    f1  = f1_score(Y[idx].cpu(), pred[idx].cpu())
    acc = accuracy_score(Y[idx].cpu(), pred[idx].cpu())
    dp, eo = fair_metric(
        pred[idx].cpu().numpy(),
        Y[idx].cpu().numpy(),
        data.sens[idx].cpu().numpy()
    )
    return auc, f1, acc, dp, eo
