# FreqLens — Paper Reproduction (under review)

This folder is the **self-contained code for GitHub**: only what is needed to reproduce the paper. Run everything from **this directory** (`paper_release/`).

## Seeds Used in the Paper

```
42, 123, 456, 789, 2024
```

## Quick Start

### 1. Environment

```bash
cd paper_release
pip install -r requirements.txt
```

### 2. Data

Set your dataset root (required):

```bash
export FREQLENS_DATA_PATH=/path/to/your/all_six_datasets
```

Expected layout: `ETT-small/`, `weather/`, `traffic/`, `electricity/` under that path. See `DATA_SETUP.md`.

### 3. Run (this package only)

**Single run (e.g. weather, seed 42):**

```bash
python train.py --dataset weather --seq_len 96 --pred_len 96 --seed 42 --epochs 50 --gpu 0
```

**All paper runs (7 datasets × 2 pred_len × 5 seeds):**

```bash
python run_all.py --gpu 0
```

Results go to `experiments/` in this folder (e.g. `experiments/v4_weather_seq96_pred96_seed42/results/test_metrics.json`).

### 4. Verify reproducibility (same seed → same result)

To confirm that the same seed gives the same result with this package:

1. Run one experiment:  
   `python train.py --dataset weather --seq_len 96 --pred_len 96 --seed 42 --epochs 50 --gpu 0`
2. Run:  
   `python check_repro.py`  
   This compares the current run to `ref_weather_seq96_pred96_seed42.json` (a prior run of this package with seed 42). Re-run step 1 and step 2 to verify determinism.

## Paper Hyperparameters (FreqLens)

| Argument | Value |
|----------|--------|
| `--seq_len` | 96 |
| `--pred_len` | 96 or 192 |
| `--d_model` | 64 |
| `--n_freq_bases` | 32 |
| `--top_k` | 8 |
| `--epochs` | 50 |
| `--batch_size` | 32 (16 for traffic, electricity) |
| `--lr` | 1e-3 |
| `--patience` | 10 |
| `--lambda_diversity` | 0.01 |
| `--lambda_recon` | 0.1 |
| `--lambda_sparse` | 0.01 |
| `--lambda_variance` | 0.1 |
| `--freq_lr_mult` | 5.0 |
| `--scheduler` | cosine |

Seeds: `42, 123, 456, 789, 2024`.

Full command template for one run (from this directory):

```bash
python train.py --dataset DATASET --seq_len 96 --pred_len PRED_LEN --seed SEED \
  --epochs 50 --batch_size 32 --gpu 0
```

Replace `DATASET` (e.g. `weather`, `traffic`, `ETTh1`), `PRED_LEN` (96 or 192), and `SEED` (42, 123, 456, 789, 2024).

## Aggregating Results (Mean ± Std)

After runs finish:

```bash
python aggregate_results.py
```

## Folder Contents

- `train.py` — single run (train from scratch; test uses final model)
- `run_all.py` — all paper experiments (skip existing)
- `check_repro.py` — compare current run to ref (verify same seed → same result)
- `ref_weather_seq96_pred96_seed42.json` — reference from a prior run of this package (seed 42)
- `aggregate_results.py` — mean ± std table
- `DATA_SETUP.md` — data layout
- `requirements.txt` — Python dependencies
- `.gitignore` — ignore `experiments/` and `*.pt` when committing

## Optional (for a standalone GitHub repo)

- **LICENSE** — e.g. MIT; add if you publish this folder as its own repo.
- **Citation** — add a BibTeX block in README if the paper is accepted.
- **Python version** — recommend 3.8+ in README if you want to be explicit.
