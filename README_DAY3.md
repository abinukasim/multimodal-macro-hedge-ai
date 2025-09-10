
# Day 3 — Baseline model + Walk-forward backtest

## Install deps
```bash
source .venv/bin/activate
pip install lightgbm scikit-learn
```

## Train & backtest (time-only vs fused)
```bash
python -m scripts.train_baseline --start-idx 252 --cost-bps 1
```

This saves:
- `data/processed/wf_time_only.parquet`, `wf_fused.parquet` (daily probabilities & signals)
- `data/processed/curve_time_only.parquet`, `curve_fused.parquet` (equity curves)

Quick check:
```bash
python -m scripts.run_day3_quicktest.py
```

## Notes
- Features = all columns except {date, y, ohlc, volume}. The **fused** run also includes FinBERT features.
- Threshold is 0.5 for a long/flat strategy. You can tune it later or use probability sizing.
- Costs are applied at 1bp per position change by default.
