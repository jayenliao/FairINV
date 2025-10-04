# logger.py
from __future__ import annotations
import csv, json, os, time
from pathlib import Path
from typing import Dict, Any, Optional
from tensorboardX import SummaryWriter

class EpochLogger:
    """
    Writes:
      • TensorBoard scalars under tags like "train/loss", "val/auc", "test/dp", ...
      • CSV with one row per epoch
      • JSONL with one object per epoch (optional but handy for post-hoc parsing)
    """
    def __init__(self, run_dir, *, model:str, write_jsonl: bool = True):
        self._t0 = time.time()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.run_dir))
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self._csv_header_written = False
        self._write_jsonl = write_jsonl
        self._csv_fieldnames = ["timestamp", "epoch", "split"]
        self._csv_fieldnames += ["acc", "auc", "f1", "dp", "eo"]
        if model == "vanilla":
            self._csv_fieldnames += ["loss_all", "l1"]
        elif model == "fairinv":
            self._csv_fieldnames += ['loss_all', 'loss_cls', 'loss_irm', 'loss_cls_all']
        elif model == "edge_adder":
            self._csv_fieldnames += ["loss_all", "loss_bce", "loss_dp", "l1"]

    def log(self, epoch: int, split: str, metrics: Dict[str, Any]):
        """
        split ∈ {"train","val","test"}; metrics like {"loss":..., "auc":..., "f1":..., "dp":..., "eo":...}
        """
        # --- TensorBoard ---
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                self.tb.add_scalar(f"{split}/{k}", float(v), epoch)
            except Exception:
                pass  # ignore non-numerics

        # --- CSV ---
        row = {"epoch": epoch, "split": split, **metrics}
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **row  # keep original keys from caller; they may contain 'split','epoch','acc', etc.
        }
        # Normalize to the fixed header: fill missing keys with '' so DictWriter never changes columns.
        normalized = {k: row.get(k, "") for k in self._csv_fieldnames}

        # Always use the same fieldnames
        with self.csv_path.open("a", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=self._csv_fieldnames)
            if not self._csv_header_written or self.csv_path.stat().st_size == 0:
                w.writeheader()
                self._csv_header_written = True
            w.writerow(normalized)

        # --- JSONL (optional) ---
        if self._write_jsonl:
            with self.jsonl_path.open("a") as jf:
                json.dump(row, jf, ensure_ascii=False)
                jf.write("\n")

    def close(self):
        self.tb.flush()
        self.tb.close()
