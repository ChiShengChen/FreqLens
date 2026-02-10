#!/usr/bin/env python3
"""
Verify that paper_release reproduces the same result with the same seed.
Compares current run to a reference from a prior run of THIS package (paper_release) with seed 42.

Usage: run  python train.py --dataset weather --seq_len 96 --pred_len 96 --seed 42 --epochs 50 --gpu 0
       then  python check_repro.py
Reference (ref_weather_seq96_pred96_seed42.json) = output of a previous paper_release run with seed 42.
"""

import json
from pathlib import Path

PROJ = Path(__file__).resolve().parent
REF = PROJ / "ref_weather_seq96_pred96_seed42.json"
EXP_DIR = PROJ / "experiments" / "v4_weather_seq96_pred96_seed42"
METRICS = EXP_DIR / "results" / "test_metrics.json"


def main():
    if not REF.is_file():
        print("Reference not found:", REF)
        return 1
    ref = json.loads(REF.read_text())
    if not METRICS.is_file():
        print("No result yet. Run first:")
        print("  python train.py --dataset weather --seq_len 96 --pred_len 96 --seed 42 --epochs 50 --gpu 0")
        return 1
    new = json.loads(METRICS.read_text())
    tol = 1e-5
    mse_ok = abs(new["MSE"] - ref["MSE"]) < tol
    mae_ok = abs(new["MAE"] - ref["MAE"]) < tol
    print("Reference (prior paper_release run, seed 42):  MSE =", ref["MSE"], " MAE =", ref["MAE"])
    print("Current run (paper_release, seed 42):          MSE =", new["MSE"], " MAE =", new["MAE"])
    print("Match (tolerance 1e-5):", "YES" if (mse_ok and mae_ok) else "NO")
    return 0 if (mse_ok and mae_ok) else 1


if __name__ == "__main__":
    exit(main())
