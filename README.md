
# Multimodal Macro Hedge Fund AI — Starter (Week 1)

This repo is a *minimal* scaffold to get you from zero → **Day 1** working data pipeline.

## What you’ll have after Day 1
- A local Python environment.
- A repo with clean structure.
- Code that fetches **SPY**, **VIX**, and **FRED** 3M/10Y yields.
- A script that writes a merged `market.csv` to `data/processed/`.
- A sanity check run (`python run_day1_quicktest.py`).

---

## Prereqs (macOS)
- Python 3.10+ (check with `python3 --version`).
- VS Code installed (with the Python extension).
- Git + GitHub account.

*(Windows: use `py -m venv .venv` and `.venv\Scripts\activate` instead of the macOS/Linux commands.)*

---

## Setup (Day 1)
```bash
# 1) Clone or unzip this starter
cd multimodal-macro-hedge-ai

# 2) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Upgrade pip and install deps
pip install --upgrade pip
pip install -r requirements.txt

# 4) Sanity check: fetch data and write CSVs
python scripts/bootstrap_data.py

# 5) Quick test
python run_day1_quicktest.py
```

If everything works, you should see a preview of the merged market dataframe and a file at `data/processed/market.csv` (and `market_head.csv`).

---

## Next (Day 2 preview)
- Add a headlines CSV to `data/raw/headlines.csv` with columns: `date,headline`.
- You'll build **text features** with FinBERT and join them to market data.
- Then you'll train your first baseline model with walk-forward validation.

---

## Project structure
```
multimodal-macro-hedge-ai/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/
│  └─ processed/
├─ configs/
│  └─ experiment.yaml
├─ notebooks/
├─ scripts/
│  └─ bootstrap_data.py
├─ src/
│  ├─ data/
│  │  └─ fetch_market.py
│  ├─ features/
│  │  └─ ts_features.py
│  ├─ models/        # (coming Day 4-5)
│  ├─ nlp/           # (coming Day 3)
│  ├─ backtest/      # (coming Day 5)
│  └─ utils/
├─ app/
│  └─ streamlit_app.py
└─ run_day1_quicktest.py
```
