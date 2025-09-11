# Macro Multimodal Hedge AI

*A research prototype that fuses financial **text** + **market time-series** to forecast next-day S&P 500 direction and backtest trading rules with costs. Ships with a live Streamlit dashboard and a daily automation pipeline.*

---

## TL;DR

- **Modalities:** Official macro texts (Fed statements/minutes & speeches, BLS CPI & NFP notes, BEA PCE) + market features (SPY, VIX, 10Y, 3M, term-structure, returns, vols).
- **Model:** FinBERT sentence embeddings aggregated daily → concatenated with engineered time-series features → **XGBoost** classifier → probability → **anchored isotonic calibration** → position sizing.
- **Backtest (net of 1bp costs)** — recent runs:
  - **Time-only:** Sharpe ≈ **0.59**, MaxDD ≈ **−0.27**
  - **Fused (biweekly refit, prob sizing, anchored calibration):** Sharpe ≈ **0.74–0.76**, MaxDD ≈ **−0.23**
- **App:** `/app/streamlit_app.py` shows equity curves, metrics, and latest signals.
- **Ops:** `scripts/daily_refresh.sh` fetches new text, rebuilds features, retrains, calibrates, and refreshes artifacts (cron/launchd examples included).

> ⚠️ Research only. Not investment advice. Historical performance ≠ future results.

---

## Why this project

Macro markets react to *words* (policy guidance, data releases) and *numbers* (vol, term spread). This project builds a minimal, pragmatic pipeline that fuses both streams, emphasizes **walk-forward validation**, and reports **cost-aware** backtests you can interrogate live in a dashboard.

---

## Repo structure

multimodal-macro-hedge-ai/

├─ app/

│  └─ streamlit_app.py         # Dashboard (equity curves, metrics, latest signals)

├─ configs/                    # Optional config files

├─ data/

│  ├─ raw/                     # Ingested text csvs (ignored by git)

│  └─ processed/               # Parquet artifacts (ignored by git)

├─ notebooks/                  # Scratch/EDA

├─ scripts/

│  ├─ bootstrap_data.py        # Fetch market data (offline fallback)

│  ├─ build_text_features.py   # FinBERT embeddings → daily features

│  ├─ build_fusion_dataset.py  # Join text + time features → model table

│  ├─ train_baseline.py        # Walk-forward training & backtest

│  ├─ calibrate_probs.py       # Anchored isotonic calibration + curve

│  ├─ fetch_official_text.py   # RSS scrapers for Fed/BLS/BEA

│  ├─ ingest_text_sources.py   # Merge multiple text csvs → headlines.csv

│  └─ daily_refresh.sh         # End-to-end refresh pipeline (cron-safe)

├─ src/

│  ├─ backtest/                # PnL curve, turnover/costs, metrics

│  ├─ data/                    # Market fetchers/parsers

│  ├─ features/                # Time-series feature engineering

│  ├─ models/                  # Walk-forward utilities (XGBoost)

│  └─ utils/                   # Helpers

├─ requirements.txt

└─ README.md



---
## Quick start

```bash
# 1) create & activate venv
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

# 2) bootstrap market data (falls back to offline synthetic if remote fails)
python -m scripts.bootstrap_data

# 3) fetch recent official texts (RSS) and build features
python -m scripts.fetch_official_text --since 2018-01-01 --outdir mytexts
python -m scripts.ingest_text_sources mytexts/*.csv --out data/raw/headlines.csv
python -m scripts.build_text_features --input data/raw/headlines.csv

# 4) fuse modalities and train (biweekly refits worked best in tests)
python -m scripts.build_fusion_dataset
python -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 10 --cost-bps 1

# 5) leakage-safe calibration (anchor cut)
python -m scripts.calibrate_probs --cut 2023-01-01 --sizing prob --prob-scale 0.06

# 6) run dashboard
streamlit run app/streamlit_app.py
---

**Artifacts of interest**

* `data/processed/text_features.parquet` — daily FinBERT features
* `data/processed/fusion_dataset.parquet` — model table
* `data/processed/wf_*.parquet` — walk-forward predictions
* `data/processed/wf_fused_cal.parquet` — calibrated live window (≥ cut)
* `data/processed/curve_fused.parquet` — **dashboard curve**

---

## Modeling details

**Text pipeline**

* Sources: Fed statements/minutes & speeches, BLS CPI & Employment Situation notes, BEA PCE (via official RSS).
* Preprocess: strip boilerplate; keep headline + first paragraph; date-align to the market day.
* Embeddings: **FinBERT** (HuggingFace) → mean pooling → daily aggregate.
* Sentiment features: probabilities *(pos/neu/neg)* per day.

**Time-series features**

* Price & returns (SPY), realized vol, VIX, 10Y and 3M yields, term spread, rolling z-scores, and lags.

**Learner**

* **XGBoost** binary classifier predicting  **P(up tomorrow)** .
* **Walk-forward** training:
  * Rolling window; holdout = next day.
  * Refit cadence: **every 10 days** (empirically robust & faster).
  * Metrics: AUC, Accuracy, Hit-rate.

**Calibration & sizing**

* **Anchored isotonic calibration** : fit on a pre-defined *calibration window* (e.g., < 2023-01-01), apply to the live period only (no leakage).
* **Position sizing:** probabilistic

  post=clip(pt−0.5scale,0,1)\text{pos}_t=\text{clip}\Big(\frac{p_t-0.5}{\text{scale}}, 0, 1\Big)**pos**t****=**clip**(**scale**p**t****−**0.5****,**0**,**1**)
  with scale ≈  **0.06** ; optional binary/hysteresis supported.
* **Transaction costs:** default **1 bp** per trade; configurable.

**Backtest**

* Daily rebalancing with cost-aware turnover.
* Report: Net Return, CAGR,  **Sharpe (net)** ,  **Max Drawdown** , trades, average turnover.

---

## Results (illustrative from recent runs)

| Setup                                                                | AUC              | Sharpe (net)         | MaxDD             | Trades    |
| -------------------------------------------------------------------- | ---------------- | -------------------- | ----------------- | --------- |
| Time-only (step=10, 1bp)                                             | ~0.512           | **0.59**       | −0.27            | ~580      |
| **Fused**(step=10, 1bp, prob scale=0.06, anchored calibration) | **~0.516** | **0.74–0.76** | **~−0.23** | ~560–760 |

*Values vary slightly with data refresh; the dashboard shows the latest.*

---

## Dashboard

* **Equity curves** (net of costs),  **Key metrics** , **Latest signals**
* Start with:
  <pre class="overflow-visible!" data-start="5958" data-end="6008"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>streamlit run app/streamlit_app.py
  </span></span></code></div></div></pre>
* (Optional) Show current config by reading `data/processed/fused_config.json` in the app header.

---

## Automation (daily refresh)

Rebuild everything end-to-end (cron-safe):

<pre class="overflow-visible!" data-start="6188" data-end="6226"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>./scripts/daily_refresh.sh
</span></span></code></div></div></pre>

Example cron entry (macOS):

<pre class="overflow-visible!" data-start="6256" data-end="6421"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-cron"><span>10 8 * * 1-5 /bin/bash "/absolute/path/multimodal-macro-hedge-ai/scripts/daily_refresh.sh" >> "/absolute/path/multimodal-macro-hedge-ai/refresh.log" 2>&1
</span></code></div></div></pre>

> macOS tip: if using cron, grant Full Disk Access to your terminal app. If your Mac sleeps at run time, consider `launchd`.

---

## Design choices & lessons

* **Anchored calibration** stabilizes probability scales across regimes.
* **Biweekly refit** improved stability/Sharpe vs. daily refits in tests.
* A modest **prob-scale** or **no-trade band/hysteresis** reduces churn and costs.
* RSS gives *recent* text only; deeper archives should further improve coverage.

---

## Limitations

* **Data coverage:** RSS is not full-history; archive scrapers (Fed/BLS/BEA listings) are on the roadmap.
* **Single asset:** Currently SPY; multi-asset extension is straightforward.
* **Model class:** XGBoost baseline; sequence models (TFT, TCN, transformers) are future work.
* **Live trading:** This is a  *research tool* , not a production trading system.

---

## Roadmap

* Add **archive scrapers** for full 2018→today text coverage.
* Try **temporal models** (TFT/TCN) and **feature selection** with SHAP.
* Regime-aware sizing (vol targeting, drawdown caps).
* News/RSS expansion (Treasury, ISM, FOMC transcripts, SEC 10-K MD&A).
* Cross-asset (rates, credit, FX, commodities) + sector/tilt overlays.

---

## Setup notes

* Python 3.12, `xgboost`, `pandas`, `numpy`, `scikit-learn`, `transformers`, `streamlit`.
* macOS tip: if using LightGBM you’ll need `libomp`; this project defaults to **XGBoost** to avoid that dependency.

---

## Scripts cookbook

<pre class="overflow-visible!" data-start="7892" data-end="8521"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span># Market data (offline fallback built-in)</span><span>
python -m scripts.bootstrap_data

</span><span># Text → features</span><span>
python -m scripts.fetch_official_text --since 2018-01-01 --outdir mytexts
python -m scripts.ingest_text_sources mytexts/*.csv --out data/raw/headlines.csv
python -m scripts.build_text_features --input data/raw/headlines.csv

</span><span># Fuse, train, calibrate, backtest</span><span>
python -m scripts.build_fusion_dataset
python -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 10 --cost-bps 1
python -m scripts.calibrate_probs --</span><span>cut</span><span> 2023-01-01 --sizing prob --prob-scale 0.06

</span><span># Dashboard</span><span>
streamlit run app/streamlit_app.py
</span></span></code></div></div></pre>

---

## .gitignore (recommended)

<pre class="overflow-visible!" data-start="8557" data-end="8675"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>.venv/
__pycache__/
*.</span><span>log</span><span>
data/raw/**
data/processed/**
mytexts/**
!data/raw/.gitkeep
!data/processed/.gitkeep
</span></span></code></div></div></pre>

---

## Disclaimer & License

This repository is for **research and educational purposes only** and does **not** constitute investment advice or an offer to trade. Use at your own risk.

License: MIT
