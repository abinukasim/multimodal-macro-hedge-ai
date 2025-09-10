# src/data/fetch_market.py
import time
import pandas as pd
import numpy as np
import yfinance as yf

# ---------- helpers ----------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index, standardize column names, add naive 'date'."""
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        if "index" in df.columns:
            df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

def _yf_hist(ticker: str, start: str) -> pd.DataFrame:
    """Robust single-ticker download via Ticker.history with retries."""
    last_err = None
    for i in range(3):
        try:
            t = yf.Ticker(ticker)
            df = t.history(
                start=start,
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize(df)
        except Exception as e:
            last_err = e
        time.sleep(1 + i)  # simple backoff
    raise RuntimeError(f"yfinance history failed for {ticker}: {last_err}")

# ---------- fetchers ----------
def get_prices(symbol="SPY", start="2010-01-01") -> pd.DataFrame:
    return _yf_hist(symbol, start)

def get_vix(start="2010-01-01", spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    # Try ^VIX, else realized-vol proxy from SPY so the pipeline never breaks
    try:
        vix = _yf_hist("^VIX", start)[["date", "close"]].rename(columns={"close": "vix"})
        return vix
    except Exception:
        if spy_df is None or spy_df.empty:
            return pd.DataFrame({"date": pd.to_datetime([]), "vix": []})
        tmp = spy_df[["date", "close"]].copy()
        tmp["ret"] = tmp["close"].pct_change()
        rv = tmp["ret"].rolling(21).std() * np.sqrt(252) * 100.0  # annualized %
        out = tmp[["date"]].copy()
        out["vix"] = rv
        return out

def get_yield_10y(start="2010-01-01") -> pd.DataFrame:
    # In your probe, ^TNX already returned ~4.03 (percent). Use it directly.
    try:
        tnx = _yf_hist("^TNX", start)[["date", "close"]]
        tnx["dgs10"] = tnx["close"]
        return tnx[["date", "dgs10"]]
    except Exception:
        return pd.DataFrame({"date": pd.to_datetime([]), "dgs10": []})

def get_yield_3m(start="2010-01-01") -> pd.DataFrame:
    try:
        irx = _yf_hist("^IRX", start)[["date", "close"]]
        irx["dgs3mo"] = irx["close"]
        return irx[["date", "dgs3mo"]]
    except Exception:
        return pd.DataFrame({"date": pd.to_datetime([]), "dgs3mo": []})

def merge_market(start="2010-01-01") -> pd.DataFrame:
    spy = get_prices("SPY", start)
    vix = get_vix(start, spy_df=spy)
    dgs10 = get_yield_10y(start)
    dgs3m = get_yield_3m(start)

    m = (
        spy.merge(vix, on="date", how="left")
           .merge(dgs10, on="date", how="left")
           .merge(dgs3m, on="date", how="left")
    )

    for col in ["dgs10", "dgs3mo", "vix"]:
        if col in m.columns:
            m[col] = m[col].ffill()

    return m