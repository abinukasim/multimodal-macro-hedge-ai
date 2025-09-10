import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # standardize all column names to lowercase once
    df.columns = [c.lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError("Expected a 'close' column in the input DataFrame.")

    # base returns/vol
    df["ret1"] = df["close"].pct_change()
    for k in [2, 5, 10, 20]:
        df[f"ret{k}"] = df["close"].pct_change(k)
        df[f"vol{k}"] = df["ret1"].rolling(k).std()

    # vix change (if present)
    if "vix" in df.columns:
        df["vix_chg"] = df["vix"].pct_change()
    else:
        df["vix_chg"] = np.nan  # keep pipeline working even without vix

    # term spread (10y - 3m) if both present
    if "dgs10" in df.columns and "dgs3mo" in df.columns:
        df["term_spread"] = df["dgs10"] - df["dgs3mo"]
    else:
        df["term_spread"] = np.nan

    # drop only the rows that are invalid for *core* features
    must_have = ["ret1", "ret2", "ret5", "ret10", "vol5", "vol10", "vol20"]
    df = df.dropna(subset=[c for c in must_have if c in df.columns])

    return df