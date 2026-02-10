"""
FreqLens training (paper reproduction). Train from scratch; test uses final model (no loading checkpoint).
Run from this directory:  python train.py --dataset weather --seq_len 96 --pred_len 96 --seed 42 --gpu 0
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import torch
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from models.freqlens_v4 import FreqLensV4
from data.data_loader import get_dataloader
from trainers.forecasting_trainer import ForecastingTrainer
from utils.metrics import evaluate_metrics


def parse_args():
    p = argparse.ArgumentParser(description="FreqLens training (paper settings, train from scratch)")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "weather", "traffic"])
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"])
    p.add_argument("--scale", action="store_true", default=True)
    p.add_argument("--use_paper_splits", action="store_true", default=False)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_freq_bases", type=int, default=32)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freq_init", type=str, default="log_uniform", choices=["log_uniform", "linear", "random"])
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--lambda_diversity", type=float, default=0.01)
    p.add_argument("--lambda_recon", type=float, default=0.1)
    p.add_argument("--lambda_sparse", type=float, default=0.01)
    p.add_argument("--lambda_variance", type=float, default=0.1)
    p.add_argument("--lambda_ortho", type=float, default=0.0)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--freq_lr_mult", type=float, default=5.0)
    p.add_argument("--freeze_frequencies", action="store_true", default=False)
    p.add_argument("--no_residual_path", action="store_true", default=False)
    p.add_argument("--shared_attribution", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="experiments")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu: int) -> str:
    if gpu < 0 or not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{gpu}"


def analyze_frequencies(model, dataset, exp_dir, device):
    model.eval()
    granularity = {
        "weather": {"unit": "10min", "steps_per_hour": 6},
        "traffic": {"unit": "hourly", "steps_per_hour": 1},
        "electricity": {"unit": "hourly", "steps_per_hour": 1},
        "ETTh1": {"unit": "hourly", "steps_per_hour": 1},
        "ETTh2": {"unit": "hourly", "steps_per_hour": 1},
        "ETTm1": {"unit": "15min", "steps_per_hour": 4},
        "ETTm2": {"unit": "15min", "steps_per_hour": 4},
    }
    known_periods_hours = {"half_daily": 12.0, "daily": 24.0, "weekly": 168.0, "monthly": 720.0, "yearly": 8760.0}
    ds_info = granularity.get(dataset, {"steps_per_hour": 1})
    sph = ds_info["steps_per_hour"]
    with torch.no_grad():
        freqs = model.freq_decomp.frequencies.cpu().numpy()
        periods_steps = 1.0 / (freqs + 1e-12)
        periods_hours = periods_steps / sph
    order = np.argsort(periods_hours)
    matches = []
    for i in order:
        ph = periods_hours[i]
        pd_val = ph / 24.0
        best_name, best_err = None, float("inf")
        for name, kp in known_periods_hours.items():
            err = abs(ph - kp) / kp
            if err < best_err:
                best_err, best_name = err, name
        matches.append({
            "freq_idx": int(i), "frequency": float(freqs[i]), "period_steps": float(periods_steps[i]),
            "period_hours": float(ph), "period_days": float(pd_val), "closest_known": best_name,
            "relative_error": float(best_err), "is_match": bool(best_err < 0.20),
        })
    os.makedirs(os.path.join(exp_dir, "frequency_analysis"), exist_ok=True)
    analysis = {
        "dataset": dataset, "time_unit": ds_info.get("unit", "unknown"), "steps_per_hour": sph,
        "n_frequencies": len(freqs), "matches": matches,
        "n_matched": sum(1 for m in matches if m["is_match"]),
        "n_novel": sum(1 for m in matches if not m["is_match"]),
    }
    path = os.path.join(exp_dir, "frequency_analysis", "learned_frequencies.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    return analysis


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu)
    exp_name = args.exp_name or f"v4_{args.dataset}_seq{args.seq_len}_pred{args.pred_len}_seed{args.seed}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    loader_kw = dict(
        dataset_name=args.dataset, seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, num_workers=args.num_workers,
        label_len=args.label_len, features=args.features, scale=args.scale,
        use_paper_splits=args.use_paper_splits,
    )
    train_loader = get_dataloader(split="train", **loader_kw)
    val_loader = get_dataloader(split="val", **loader_kw)
    test_loader = get_dataloader(split="test", **loader_kw)
    enc_in = next(iter(train_loader))[0].shape[-1]

    model = FreqLensV4(
        seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in,
        d_model=args.d_model, n_freq_bases=args.n_freq_bases, top_k=args.top_k,
        dropout=args.dropout, freq_init=args.freq_init, gumbel_tau=args.gumbel_tau,
        freeze_frequencies=args.freeze_frequencies, no_residual=args.no_residual_path,
        shared_attribution=args.shared_attribution,
    ).to(device)
    n_params = model.count_parameters()

    freq_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "log_frequencies" in name or "freq_decomp.phases" in name:
            freq_params.append(param)
        else:
            other_params.append(param)
    if freq_params:
        optimizer = torch.optim.Adam([
            {"params": freq_params, "lr": args.lr * args.freq_lr_mult},
            {"params": other_params, "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.Adam(other_params, lr=args.lr)

    trainer_config = {
        "num_epochs": args.epochs, "lambda_sparse": args.lambda_sparse, "lambda_recon": args.lambda_recon,
        "lambda_ortho": args.lambda_ortho, "lambda_variance": args.lambda_variance,
        "lambda_diversity": args.lambda_diversity, "scheduler": args.scheduler, "min_lr": args.min_lr,
        "early_stopping_patience": args.patience, "checkpoint_dir": os.path.join(exp_dir, "checkpoints"),
        "gumbel_tau_init": args.gumbel_tau, "gumbel_tau_min": 0.1, "gradient_clip": args.gradient_clip,
        "log_interval": args.log_interval, "save_interval": args.save_interval,
    }
    trainer = ForecastingTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, config=trainer_config, device=device,
    )

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0
    # Test with final model only (train from scratch; do not load best checkpoint)
    test_results = trainer.predict(test_loader)
    metrics = test_results["metrics"]

    def to_native(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_native(x) for x in obj]
        return obj

    with open(os.path.join(exp_dir, "results", "test_metrics.json"), "w") as f:
        json.dump(to_native(metrics), f, indent=2)
    with open(os.path.join(exp_dir, "results", "training_history.json"), "w") as f:
        json.dump(to_native(history), f, indent=2)
    trainer.save_checkpoint(os.path.join(exp_dir, "checkpoints", "final_model.pt"), args.epochs)
    analyze_frequencies(model, args.dataset, exp_dir, device)

    print(f"MSE={metrics['MSE']:.6f}  MAE={metrics['MAE']:.6f}  Time={elapsed/60:.1f}min  Dir={exp_dir}")


if __name__ == "__main__":
    main()
