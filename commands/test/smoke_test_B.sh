#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash smoke_test_B.sh                # default: dataset=german, model=vanilla
#   bash smoke_test_B.sh pokec_z vanilla
#   bash smoke_test_B.sh german edge_adder

DATASET="${1:-german}"
MODEL="${2:-vanilla}"

# Decide device
HAS_CUDA="$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
DEVICE="cuda"
if [[ "$HAS_CUDA" != "1" ]]; then
  DEVICE="cpu"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOGROOT="logs/test/smoke_B_${TS}"
mkdir -p "$LOGROOT"

COMMON=(
  --dataset "$DATASET"
  --model "$MODEL"
  --seed_num 5
  --start_seed 42
  --epochs 100
  --log_interval 1
  --num_threads 1
  --device "$DEVICE"
  --log_dir "$LOGROOT"
)

echo "============================================================"
echo "[1/2] Clean run (no attack) | dataset=$DATASET model=$MODEL device=$DEVICE"
echo "Log root: $LOGROOT"
echo "============================================================"
python train.py "${COMMON[@]}" --attack none | tee "$LOGROOT/clean.out"

echo
echo "============================================================"
echo "[2/2] B run (eval-only NIFA) | attack_when=eval"
echo "============================================================"
python train.py "${COMMON[@]}" \
  --attack nifa \
  --attack_when eval \
  --nifa_mode degree \
  --nifa_node 6 \
  --nifa_edge 10 \
  --nifa_loops 2 \
  --nifa_epochs 20 \
  | tee "$LOGROOT/b_eval.out"

echo
echo "============================================================"
echo "[Check] Ensure NO training-time injection happened in B run"
echo "============================================================"
if grep -q "Applying node+edge injection attack before training" "$LOGROOT/b_eval.out"; then
  echo "❌ FAIL: Found training-time injection message. B is NOT eval-only."
  exit 1
fi
echo "✅ PASS: No training-time injection message in B run."

echo
echo "============================================================"
echo "[Check] Try to find *_clean keys in any jsonl logs (optional but recommended)"
echo "============================================================"
python - <<'PY' "$LOGROOT"
import sys, os, glob, json

logroot = sys.argv[1]
seed_dirs = glob.glob(os.path.join(logroot, "**", "seed_*"), recursive=True)
if not seed_dirs:
    print(f"⚠️  Cannot find seed_* under {logroot}. Your log structure may differ.")
    sys.exit(0)

seed_dir = max(seed_dirs, key=os.path.getmtime)
print("Seed dir:", seed_dir)

jsonl_files = glob.glob(os.path.join(seed_dir, "*.jsonl")) + glob.glob(os.path.join(seed_dir, "**", "*.jsonl"), recursive=True)
found = False
for fp in jsonl_files:
    try:
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if any(k.endswith("_clean") for k in obj.keys()):
                    ks = sorted([k for k in obj.keys() if k.endswith("_clean")])
                    print(f"✅ Found *_clean keys in: {fp}")
                    print("   keys:", ks[:20], ("..." if len(ks) > 20 else ""))
                    found = True
                    break
    except Exception:
        continue
    if found:
        break

if not found:
    print("⚠️  Didn't find *_clean keys in jsonl logs.")
    print("    (Could be normal if your logger uses another format/name;")
    print("     B can still be correct as long as eval-only injection is confirmed.)")

# Also print results json if exists
res = glob.glob(os.path.join(os.path.dirname(seed_dir), "results_among_*_seeds.json"))
if res:
    rp = res[0]
    print("Results file:", rp)
    with open(rp, "r") as f:
        d = json.load(f)
    for k in ["AUC_mean","F1_mean","ACC_mean","DP_mean","EO_mean"]:
        if k in d:
            print(f"  {k}: {d[k]}")
PY

echo
echo "Done. Logs saved at: $LOGROOT"
