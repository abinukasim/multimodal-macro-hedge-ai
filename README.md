# Macro Multimodal Hedge AI

A compact research project that fuses **financial text** (Fed/BLS/BEA) with **market time-series** to forecast next-day S&P 500 direction, backtest cost-aware strategies, and visualize results in a Streamlit dashboard.

> Research only. Not investment advice.

---

## What it does

- Scrapes recent **official macro text** via RSS (Fed statements & speeches, BLS CPI & NFP notes, BEA PCE).
- Builds **FinBERT** sentiment features and joins them with **market features** (SPY, VIX, 10Y, 3M, term spread, lags, vols).
- Trains an **XGBoost** classifier in a **walk-forward** loop to predict P(up tomorrow).
- Applies **anchored isotonic calibration** and converts probabilities to positions (prob-scaled or binary).
- Backtests with **transaction costs** and shows curves/metrics in a dashboard.

---

## Repo layout



multimodal-macro-hedge-ai/

├─ app/

│  └─ streamlit_app.py                # Streamlit dashboard: equity curves, metrics, latest signals

├─ configs/

│  └─ experiment.yaml                 # (optional) example config / notes

├─ data/                              # GENERATED ARTIFACTS (git-ignored except .gitkeep)

│  ├─ raw/

│  │  ├─ headlines.csv                # merged official-text headlines (generated)

│  │  └─ .gitkeep

│  └─ processed/

│     ├─ market.csv                   # market panel (SPY/VIX/yields) with features (generated)

│     ├─ text_features.parquet        # daily FinBERT features (generated)

│     ├─ fusion_dataset.parquet       # joined table: time-series + text features (generated)

│     ├─ wf_time_only.parquet         # walk-forward predictions (time-only model) (generated)

│     ├─ wf_fused.parquet             # walk-forward predictions (fused model) (generated)

│     ├─ wf_fused_cal.parquet         # calibrated live-window predictions (generated)

│     ├─ curve_time_only.parquet      # equity curve (time-only) net of costs (generated)

│     ├─ curve_fused.parquet          # equity curve (fused) net of costs (generated)

│     └─ sweep_results.csv            # threshold/prob-scale sweep results (generated)

├─ notebooks/

│  └─ ...                             # exploratory notebooks / EDA

├─ scripts/

│  ├─ bootstrap_data.py               # fetch market data (SPY/VIX/10Y/3M) with offline fallback

│  ├─ fetch_official_text.py          # scrapers for Fed/BLS/BEA RSS; saves CSVs

│  ├─ ingest_text_sources.py          # merge text CSVs → data/raw/headlines.csv

│  ├─ build_text_features.py          # FinBERT embeddings → daily sentiment features

│  ├─ build_fusion_dataset.py         # join text + time-series into model table

│  ├─ train_baseline.py               # walk-forward training/backtest (XGBoost)

│  ├─ calibrate_probs.py              # anchored isotonic calibration + curve export

│  ├─ sweep_thresholds.py             # grid search over thresholds / prob scales

│  ├─ run_day1_quicktest.py           # smoke test: data bootstrap sanity

│  ├─ run_day2_quicktest.py           # smoke test: text features + fusion table

│  ├─ run_day3_quicktest.py           # smoke test: walk-forward outputs

│  └─ daily_refresh.sh                # cron-safe end-to-end refresh pipeline

├─ src/

│  ├─ backtest/

│  │  ├─  **init** .py

│  │  └─ backtest.py                  # pnl_curve, turnover/costs, performance metrics

│  ├─ data/

│  │  ├─  **init** .py

│  │  └─ fetch_market.py              # yfinance/stooq helpers; merge_market(), get_prices()

│  ├─ features/

│  │  ├─  **init** .py

│  │  └─ ts_features.py               # returns, rolling vol, term spread, lags, z-scores

│  ├─ models/

│  │  ├─  **init** .py

│  │  └─ walk_forward.py              # rolling fit/predict, refit cadence, model wrapper

│  ├─ nlp/                            # (if present) text embedding helpers

│  │  ├─  **init** .py

│  │  └─ finbert.py                   # FinBERT utilities (optional)

│  └─ utils/

│     ├─  **init** .py

│     └─ io.py                        # small I/O helpers (optional)

├─ .gitignore                         # ignores .venv/, logs, data/, mytexts/, etc.

├─ requirements.txt

└─ README.md


## Quick start

```bash
# 1) setup
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

# 2) market data (has offline fallback)
python -m scripts.bootstrap_data

# 3) get official text and build text features
python -m scripts.fetch_official_text --since 2018-01-01 --outdir mytexts
python -m scripts.ingest_text_sources mytexts/*.csv --out data/raw/headlines.csv
python -m scripts.build_text_features --input data/raw/headlines.csv

# 4) fuse modalities and train (walk-forward, refit every 10 days)
python -m scripts.build_fusion_dataset
python -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 10 --cost-bps 1

# 5) leakage-safe calibration + position sizing
python -m scripts.calibrate_probs --cut 2023-01-01 --sizing prob --prob-scale 0.06

# 6) dashboard
streamlit run app/streamlit_app.py
```


**Artifacts to look at**

* `data/processed/fusion_dataset.parquet` (joined table)
* `data/processed/wf_*.parquet` (walk-forward predictions)
* `data/processed/wf_fused_cal.parquet` (calibrated window)
* `data/processed/curve_*.parquet` (equity curves for the app)



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

  pos_t = clip(((p_t-0.5)/scale), 0, 1)

  with scale ≈  **0.06** ; optional binary/hysteresis supported.
* **Transaction costs:** default **1 bp** per trade; configurable.

**Backtest**

* Daily rebalancing with cost-aware turnover.
* Report: Net Return, CAGR,  **Sharpe (net)** ,  **Max Drawdown** , trades, average turnover.

## Results snapshot (illustrative)

* Time-only (1 bp costs): Sharpe ~0.59, MaxDD ~-0.27
* Fused (prob sizing + anchored calibration): Sharpe ~0.74–0.76, MaxDD ~-0.23

Numbers vary slightly with each refresh; see the dashboard for current values.


## Dashboard

* **Equity curves** (net of costs),  **Key metrics** , **Latest signals**
* Start with:
  <pre class="overflow-visible!" data-start="5958" data-end="6008"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>streamlit run app/streamlit_app.py
  </span></span></code></div></div></pre>
* (Optional) Show current config by reading `data/processed/fused_config.json` in the app header.


## Automation (daily refresh)

Rebuild everything end-to-end (cron-safe):

<pre class="overflow-visible!" data-start="6188" data-end="6226"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>./scripts/daily_refresh.sh
</span></span></code></div></div></pre>

Example cron entry (macOS):

<pre class="overflow-visible!" data-start="6256" data-end="6421"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-cron"><span>10 8 * * 1-5 /bin/bash "/absolute/path/multimodal-macro-hedge-ai/scripts/daily_refresh.sh" >> "/absolute/path/multimodal-macro-hedge-ai/refresh.log" 2>&1
</span></code></div></div></pre>

> macOS tip: if using cron, grant Full Disk Access to your terminal app. If your Mac sleeps at run time, consider `launchd`.




## Design choices & lessons

* **Anchored calibration** stabilizes probability scales across regimes.
* **Biweekly refit** improved stability/Sharpe vs. daily refits in tests.
* A modest **prob-scale** or **no-trade band/hysteresis** reduces churn and costs.
* RSS gives *recent* text only; deeper archives should further improve coverage.


## Limitations

* **Data coverage:** RSS is not full-history; archive scrapers (Fed/BLS/BEA listings) are on the roadmap.
* **Single asset:** Currently SPY; multi-asset extension is straightforward.
* **Model class:** XGBoost baseline; sequence models (TFT, TCN, transformers) are future work.
* **Live trading:** This is a  *research tool* , not a production trading system.


## Setup notes

* Python 3.12, `xgboost`, `pandas`, `numpy`, `scikit-learn`, `transformers`, `streamlit`.
* macOS: if using LightGBM you’ll need `libomp`; this project defaults to **XGBoost** to avoid that dependency.


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
streamlit run app/streamlit_app.py</span></span></code></div></div></pre>



## .gitignore (recommended)

<pre class="overflow-visible!" data-start="8557" data-end="8675"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>.venv/
__pycache__/
*.</span><span>log</span><span>
data/raw/**
data/processed/**
mytexts/**
!data/raw/.gitkeep
!data/processed/.gitkeep</span></span></code></div></div></pre>



## License

MIT
