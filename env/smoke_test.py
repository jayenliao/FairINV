import torch, sys, os
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x.t()
    print("Matmul OK:", y.shape)

# PyG quick import checks
try:
    import torch_geometric
    from torch_scatter import scatter_mean
    from torch_sparse import SparseTensor
    from torch_geometric.data import Data
    print("PyG/Scatter/Sparse imports OK:", torch_geometric.__version__)
except Exception as e:
    print("PyG import failed:", e); sys.exit(2)

# DGL quick check
try:
    import dgl
    import torch as th
    g = dgl.graph((th.tensor([0,1,2]), th.tensor([1,2,0]))).to("cuda" if torch.cuda.is_available() else "cpu")
    print("DGL graph device:", g.device)
except Exception as e:
    print("DGL import failed:", e); sys.exit(3)

print("SMOKE TEST PASSED")
