# Data Setup for Reproduction

## Expected Directory Layout

Under one data root (e.g. `all_six_datasets/` or `all_datasets/`), place:

```
all_six_datasets/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── traffic/
│   └── traffic.csv
└── electricity/
    └── electricity.csv
```

## Setting the Data Root

The data loader in `data/data_loader.py` uses a `BASE_PATH`. By default it checks:

1. `/media/meow/Transcend/time_series_benchmark/all_datasets`
2. `/media/meow/Transcend/time_series_benchmark/all_six_datasets`

**To use your own path:** set the environment variable before running:

```bash
export FREQLENS_DATA_PATH=/path/to/your/all_six_datasets
bash reproducibility/run_paper_experiments.sh
```

The data loader in `data/data_loader.py` uses `FREQLENS_DATA_PATH` when set; otherwise it falls back to the default paths. You can also edit the `BASE_PATH` block in `data/data_loader.py` if you prefer not to use the env var.

## Dataset Sources

- **ETT**: [Electricity Transformer Temperature](https://github.com/zhouhaoyi/ETDataset) (ETT-small)
- **Weather, Traffic, Electricity**: Common benchmarks; see e.g. [Time-Series-Library](https://github.com/thuml/Time-Series-Library) or [Autoformer](https://github.com/thuml/Autoformer) data links.
