
# Day 4–6 — Thresholds, Sizing, Metrics, Dashboard

## Drop these into your repo
- Overwrite `scripts/train_baseline.py`
- Overwrite `src/backtest/backtest.py`
- Overwrite `app/streamlit_app.py`

## Re-run training with sizing controls
```bash
# Weekly refits, long-only with confidence threshold
python -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 5 --cost-bps 1 --sizing binary --threshold 0.55 --band 0.00

# Probabilistic sizing (p=0.60 => full size with prob-scale=0.10)
python -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 5 --cost-bps 1 --sizing prob --prob-scale 0.10
```

## Open the dashboard
```bash
streamlit run app/streamlit_app.py
```
