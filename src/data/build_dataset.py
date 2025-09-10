
import pandas as pd
import numpy as np

def load_market(path: str = "data/processed/market.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    return df

def load_text_features(path: str = "data/processed/text_features.parquet") -> pd.DataFrame:
    if path.endswith(".csv"):
        t = pd.read_csv(path, parse_dates=["date"])
    else:
        t = pd.read_parquet(path)
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    t.columns = [c.lower() for c in t.columns]
    return t

def build_fusion(market_df: pd.DataFrame, text_df: pd.DataFrame, fill_neutral: bool = True) -> pd.DataFrame:
    m = market_df.copy()
    t = text_df.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()

    X = m.merge(t, on="date", how="left")

    X["y"] = (X["close"].shift(-1) > X["close"]).astype(int)

    for col in ["finbert_neg", "finbert_neu", "finbert_pos"]:
        if col in X.columns and fill_neutral:
            X[col] = X[col].fillna(1/3)

    for col in [c for c in X.columns if c.startswith("emb_")]:
        X[col] = X[col].fillna(0.0)

    X = X.dropna(subset=["y"])
    return X

def save_fusion(X: pd.DataFrame, out_prefix: str = "data/processed/fusion_dataset"):
    X.to_parquet(f"{out_prefix}.parquet", index=False)
    X.to_csv(f"{out_prefix}.csv", index=False)
