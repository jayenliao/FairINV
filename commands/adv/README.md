# GNN Fairness Experiment Shell Scripts

These scripts are designed to run `train.py` with different combinations of:
- datasets: german/bail/pokec_z/pokec_n/nba
- encoders: gcn/gat/gin/sage/sgc
- models: vanilla/fairinv/edge_adder/edge_minmax
- NIFA attack timing: train/eval/both
- (optional) adv-train flags if your repo includes them

## Quick start

From repo root:
```bash
mkdir -p logs_suite
bash scripts/run_all.sh
```

## Common overrides

```bash
GPU_ID=0 EPOCHS=300 SEED_NUM=3 LOG_ROOT=./logs_suite bash scripts/run_attack_when.sh
DATASETS="bail" ENCODERS="gcn gat" MODELS="vanilla edge_minmax" bash scripts/run_defenses.sh
```

## NIFA sweep example
```bash
DATASET=bail ENCODER=gcn MODEL=vanilla NODES="32 64" EDGES="20 50" BETAS="0.5 1.0" RATIOS="0.25 0.5" MODES="uncertainty" bash scripts/run_nifa_sweep.sh
```

## If your repo supports `--advtrain`
```bash
DATASET=bail ENCODER=gcn bash scripts/run_advtrain.sh
```

If `--advtrain` is not in `train.py -h`, `run_advtrain.sh` exits with a warning.
