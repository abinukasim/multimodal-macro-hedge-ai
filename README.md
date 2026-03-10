# Macro Multimodal AI — Quant Research Dashboard

## Overview
This project builds a multimodal trading signal by combining market time-series features with macroeconomic text signals. The goal is to forecast next-day S&P 500 direction in a research-oriented, reproducible workflow that can be evaluated out-of-sample.

The modeling stack fuses classic market features (returns, volatility, rates, term spread) with NLP sentiment features extracted from official macro releases. Text signals are generated from structured ingestion and transformed into model-ready representations before fusion.

A walk-forward training pipeline with calibration is used to generate signals and evaluate strategy performance under transaction costs. Results are presented through an interactive Streamlit dashboard designed for research review and presentation.

---

## Features
- Multimodal signal (market + macro text)
- Walk-forward training
- Probability calibration and position sizing
- Backtest engine with transaction costs
- Risk metrics and diagnostics
- Interactive Streamlit dashboard
- Drawdown and rolling signal stability analysis

---

## Architecture / Pipeline

```text
Market Data + Macro Text
        ↓
Feature Engineering
        ↓
Fusion Model (XGBoost)
        ↓
Walk-Forward Training
        ↓
Probability Calibration
        ↓
Backtest Engine
        ↓
Streamlit Dashboard
```

---

## Results (Latest)

| Strategy | Net Return | CAGR | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Time-only | ~26% | ~3.4% | ~0.31 | ~-34% |
| Fused + Calibration | ~53% | ~14% | ~1.38 | ~-8% |

Fusion improves predictive signal quality by integrating macro-text context with market structure.  
Calibration then improves probability quality, which helps position sizing and materially improves risk-adjusted performance and drawdown behavior.

---

## Dashboard
The Streamlit dashboard is designed for both technical and non-technical audiences and includes:
- Equity curves
- Drawdown
- Rolling directional accuracy
- Signal diagnostics

### Screenshots
- _Placeholder:_ add `image/README/dashboard-overview.png`
- _Placeholder:_ add `image/README/dashboard-diagnostics.png`

---

## Quick Start

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Market data
```bash
python -m scripts.bootstrap_data
```

### Text ingestion
```bash
python -m scripts.fetch_official_text --since 2018-01-01 --outdir mytexts
python -m scripts.ingest_text_sources mytexts/*.csv --out data/raw/headlines.csv
python -m scripts.build_text_features --input data/raw/headlines.csv
```

### Training
```bash
python -m scripts.build_fusion_dataset
python -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 10
```

### Calibration
```bash
python -m scripts.calibrate_probs --cut 2023-01-01 --sizing prob --prob-scale 0.06
```

### Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```text
src/
  data/
  features/
  models/
  backtest/
scripts/
app/
data/
```

---

## Technologies Used
- Python
- XGBoost
- FinBERT
- Pandas / NumPy
- Streamlit
- yfinance / FRED / Stooq

---

## Future Work
- additional macro signals
- regime-aware models
- improved portfolio sizing
- multi-asset expansion
