#!/usr/bin/env python3
"""
Run all paper experiments (7 datasets × 2 pred_len × 5 seeds).
Skip runs that already have results. Usage:
  python run_all.py
  python run_all.py --quick          # 1 seed only
  python run_all.py --dataset weather
  python run_all.py --dry-run
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJ_DIR / "train.py"
SAVE_DIR = PROJ_DIR / "experiments"
ALL_SEEDS = [42, 123, 456, 789, 2024]
ALL_PRED_LENS = [96, 192]
SEQ_LEN = 96
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "traffic", "electricity"]


def exp_name(dataset: str, pred: int, seed: int) -> str:
    return f"v4_{dataset}_seq{SEQ_LEN}_pred{pred}_seed{seed}"


def is_done(dataset: str, pred: int, seed: int) -> bool:
    d = SAVE_DIR / exp_name(dataset, pred, seed) / "results" / "test_metrics.json"
    return d.is_file() or (SAVE_DIR / exp_name(dataset, pred, seed) / "test_metrics.json").is_file()


def run_one(dataset: str, pred: int, seed: int, gpu: int = 0, dry_run: bool = False) -> bool:
    name = f"{dataset}/H{pred}/s{seed}"
    if is_done(dataset, pred, seed):
        try:
            p = SAVE_DIR / exp_name(dataset, pred, seed) / "results" / "test_metrics.json"
            if not p.is_file():
                p = SAVE_DIR / exp_name(dataset, pred, seed) / "test_metrics.json"
            m = json.loads(p.read_text())
            print(f"  SKIP  {name}  MSE={m['MSE']:.6f}")
        except Exception:
            print(f"  SKIP  {name}")
        return True
    if dry_run:
        print(f"  TODO  {name}")
        return False
    print(f"  RUN   {name} ...", flush=True)
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--dataset", dataset, "--seq_len", str(SEQ_LEN), "--pred_len", str(pred), "--seed", str(seed),
        "--gpu", str(gpu), "--epochs", "50", "--batch_size", "32", "--save_dir", str(SAVE_DIR),
        "--lr", "1e-3", "--patience", "10", "--n_freq_bases", "32", "--top_k", "8", "--d_model", "64",
        "--lambda_diversity", "0.01", "--lambda_recon", "0.1", "--lambda_sparse", "0.01", "--lambda_variance", "0.1",
        "--freq_lr_mult", "5.0", "--scheduler", "cosine",
    ]
    if dataset in ("traffic", "electricity"):
        cmd[cmd.index("--batch_size") + 1] = "16"
    try:
        r = subprocess.run(cmd, cwd=str(PROJ_DIR), capture_output=True, text=True, timeout=7200, env=os.environ.copy())
        if r.returncode != 0:
            print(f"  FAIL  {name}")
            return False
        if is_done(dataset, pred, seed):
            p = SAVE_DIR / exp_name(dataset, pred, seed) / "results" / "test_metrics.json"
            if not p.is_file():
                p = SAVE_DIR / exp_name(dataset, pred, seed) / "test_metrics.json"
            m = json.loads(p.read_text())
            print(f"  OK    {name}  MSE={m['MSE']:.6f}")
            return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {name}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed only")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    seeds = [42] if args.quick else ALL_SEEDS
    datasets = [args.dataset] if args.dataset else DATASETS
    done = total = 0
    for d in datasets:
        for p in ALL_PRED_LENS:
            for s in seeds:
                total += 1
                if is_done(d, p, s):
                    done += 1
    print(f"FreqLens paper runs: {done}/{total} done")
    for d in datasets:
        for p in ALL_PRED_LENS:
            for s in seeds:
                run_one(d, p, s, gpu=args.gpu, dry_run=args.dry_run)
    print("Done. Results under experiments/")


if __name__ == "__main__":
    main()
