#!/usr/bin/env python3
"""Print mean ± std of test metrics (paper seeds). Run after experiments/."""

import json
from pathlib import Path
import numpy as np

PROJ_DIR = Path(__file__).resolve().parent
SAVE_DIR = PROJ_DIR / "experiments"
SEEDS = [42, 123, 456, 789, 2024]
PRED_LENS = [96, 192]
SEQ_LEN = 96
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "traffic", "electricity"]


def exp_name(dataset: str, pred: int, seed: int) -> str:
    return f"v4_{dataset}_seq{SEQ_LEN}_pred{pred}_seed{seed}"


def load_metrics(dataset: str, pred: int, seed: int):
    d = SAVE_DIR / exp_name(dataset, pred, seed)
    for p in (d / "results" / "test_metrics.json", d / "test_metrics.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None


def main():
    print("Dataset      H    MSE (mean ± std)        MAE (mean ± std)        N")
    print("-" * 70)
    for dataset in DATASETS:
        for pred in PRED_LENS:
            mses, maes = [], []
            for seed in SEEDS:
                m = load_metrics(dataset, pred, seed)
                if m:
                    mses.append(m["MSE"])
                    maes.append(m["MAE"])
            if mses:
                print(f"{dataset:<12} {pred:<4} {np.mean(mses):.6f} ± {np.std(mses):.6f}   {np.mean(maes):.6f} ± {np.std(maes):.6f}   {len(mses)}")
            else:
                print(f"{dataset:<12} {pred:<4} (no results)")


if __name__ == "__main__":
    main()
