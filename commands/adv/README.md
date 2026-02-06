# GNN Fairness Experiment Shell Scripts (commands/adv)

These scripts run `train.py` with different combinations of:
- datasets: german/bail/pokec_z/pokec_n/nba
- encoders: gcn/gat/gin/sage/sgc
- models: vanilla/fairinv/edge_adder/edge_minmax
- NIFA attack timing: train/eval/both
- (optional) adv-train flags if your repo includes them

They are designed to live under: `commands/adv/`
but `common.sh` will auto-detect the repo root even if you move them elsewhere.

## Quick start

From repo root:
```bash
mkdir -p commands/logs_suite
bash commands/adv/run_all.sh
```

## Common overrides

```bash
GPU_ID=0 EPOCHS=300 SEED_NUM=3 LOG_ROOT=./commands/logs_suite bash commands/adv/run_attack_when.sh
DATASETS="bail" ENCODERS="gcn gat" MODELS="vanilla edge_minmax" bash commands/adv/run_defenses.sh
```

## Import victim + NIFA hyperparameters from JSON (recommended)

Your train.py supports passing multiple JSONs to --best_overall_path (victim first, NIFA second).
Set them via env var:

```bash
BEST_OVERALL_PATHS="/path/to/victim_best_overall.json /path/to/nifa_best_overall.json" \
GPU_ID=0 EPOCHS=200 SEED_NUM=3 \
DATASETS="bail" ENCODERS="gcn" MODELS="vanilla" \
ATTACK_WHENS="train eval both" \
bash commands/adv/run_attack_when.sh
```

(Backwards compatible) Single JSON:

```bash
BEST_OVERALL_PATH="/path/to/best_overall.json" bash commands/adv/run_attack_when.sh
```

## NIFA sweep example
```bash
DATASET=bail ENCODER=gcn MODEL=vanilla \
NODES="32 64" EDGES="20 50" BETAS="0.5 1.0" RATIOS="0.25 0.5" MODES="uncertainty" \
bash commands/adv/run_nifa_sweep.sh
```

## If your repo supports `--advtrain`
```bash
DATASET=bail ENCODER=gcn bash commands/adv/run_advtrain.sh
```

If `--advtrain` is not in `train.py -h`, `run_advtrain.sh` exits with a warning.

## Troubleshooting

If needed, you can always override the train script path:

```bash
TRAIN_PY="/tmp2/jayliao/gnn_fairness/FairINV/train.py" bash commands/adv/run_attack_when.sh
```
