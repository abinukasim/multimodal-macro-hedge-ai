# src/data/fetch_market.py
import time
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

# ---------- helpers ----------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index, standardize column names, add naive 'date'."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance.download may return (field, ticker) columns.
        if len(set(df.columns.get_level_values(-1))) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            flat = []
            for col in df.columns:
                parts = [str(x) for x in col if str(x) and str(x) != "None"]
                flat.append("_".join(parts))
            df.columns = flat
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    if "date" not in df.columns:
        if "index" in df.columns:
            df = df.rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

def _yf_hist(ticker: str, start: str) -> pd.DataFrame:
    """Robust single-ticker download via yfinance with retries."""
    last_err = None
    for i in range(1):
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

        try:
            df = yf.download(
                ticker,
                start=start,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _normalize(df)
        except Exception as e:
            last_err = e

        time.sleep(1 + i)  # simple backoff
    raise RuntimeError(f"yfinance history failed for {ticker}: {last_err}")

def _stooq_hist(symbol: str, start: str) -> pd.DataFrame:
    """Stooq fallback for symbols (not all indices are available)."""
    candidates = [symbol]
    if symbol.isalpha() and "." not in symbol:
        candidates.insert(0, f"{symbol}.US")
    last_err = None
    for sym in candidates:
        try:
            df = web.DataReader(sym, "stooq", start)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out = _normalize(df)
                return out.sort_values("date").reset_index(drop=True)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"stooq failed for {symbol}: {last_err}")

def _fred_series(series: str, start: str, out_col: str) -> pd.DataFrame:
    df = web.DataReader(series, "fred", start)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame({"date": pd.to_datetime([]), out_col: []})
    out = _normalize(df)
    value_col = next((c for c in out.columns if c != "date"), None)
    if value_col is None:
        return pd.DataFrame({"date": pd.to_datetime([]), out_col: []})
    out = out.rename(columns={value_col: out_col})[["date", out_col]]
    return out.sort_values("date").reset_index(drop=True)

# ---------- fetchers ----------
def get_prices(symbol="SPY", start="2010-01-01") -> pd.DataFrame:
    try:
        return _yf_hist(symbol, start)
    except Exception:
        return _stooq_hist(symbol, start)

def get_vix(start="2010-01-01", spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    # Try ^VIX, else realized-vol proxy from SPY so the pipeline never breaks
    try:
        vix = _yf_hist("^VIX", start)[["date", "close"]].rename(columns={"close": "vix"})
        return vix
    except Exception:
        try:
            return _fred_series("VIXCLS", start, "vix")
        except Exception:
            pass
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
        return _fred_series("DGS10", start, "dgs10")

def get_yield_3m(start="2010-01-01") -> pd.DataFrame:
    try:
        irx = _yf_hist("^IRX", start)[["date", "close"]]
        irx["dgs3mo"] = irx["close"]
        return irx[["date", "dgs3mo"]]
    except Exception:
        return _fred_series("DGS3MO", start, "dgs3mo")

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
