# fresh or existing env
conda activate jayliao_gnn311

# 1) Torch 2.2.2 + cu121 and matching CUDA runtime libs
pip install --index-url https://download.pytorch.org/whl/cu121 \
  --upgrade --force-reinstall \
  "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2" \
  nvidia-nccl-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12

# 2) Stability pins that won’t mutate torch
pip install "numpy<2" "torchdata==0.6.1" --no-deps "pydantic<2"

# 3) DGL & PyG matching torch-2.2 + cu121 (NO deps so pip can’t downgrade torch)
pip install --no-deps -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html dgl
pip install --no-deps -f https://data.pyg.org/whl/torch-2.2.0+cu121.html \
  'torch-geometric==2.5.3' 'torch-scatter==2.1.2' 'torch-sparse==0.6.18' \
  'torch-cluster==1.6.3' 'torch-spline-conv==1.2.2' 'pyg-lib==0.4.0'

# 4) Prefer env’s CUDA/NCCL over the host (avoid ncclCommRegister issues)
SITE=$(python - <<'PY'
import sysconfig, pathlib; print(pathlib.Path(sysconfig.get_paths()['purelib']))
PY
)
export LD_LIBRARY_PATH="$SITE/nvidia/nccl/lib:$SITE/torch/lib:${LD_LIBRARY_PATH:-}"

# (optional) make it permanent
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/10-cuda-libs.sh" <<'SH'
SITE=$(python - <<'PY'
import sysconfig, pathlib; print(pathlib.Path(sysconfig.get_paths()['purelib']))
PY
)
export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$SITE/nvidia/nccl/lib:$SITE/torch/lib:${LD_LIBRARY_PATH:-}"
SH
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/10-cuda-libs.sh" <<'SH'
export LD_LIBRARY_PATH="${_OLD_LD_LIBRARY_PATH:-}"; unset _OLD_LD_LIBRARY_PATH
SH

# 5) Verify
python - <<'PY'
import torch, dgl, torch_geometric
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("dgl:", dgl.__version__, "pyg:", torch_geometric.__version__)
PY
