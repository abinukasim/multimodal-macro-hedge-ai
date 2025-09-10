import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from src.data.fetch_market import merge_market
from src.features.ts_features import add_time_features

def make_offline_stub(start="2010-01-01"):
    dates = pd.date_range(start, pd.Timestamp.today().normalize(), freq=BDay())
    rng = np.random.RandomState(42)
    # geometric random walk with mild drift
    ret = rng.normal(0.0004, 0.01, len(dates))
    close = 100 * (1 + pd.Series(ret, index=dates)).cumprod()
    high = close * (1 + 0.005 * rng.rand(len(dates)))
    low = close * (1 - 0.005 * rng.rand(len(dates)))
    openp = close.shift(1).fillna(close.iloc[0])
    vol = (1e7 + 2e6 * rng.rand(len(dates))).astype(int)
    vix = 20 + 5 * rng.randn(len(dates))
    term_spread = 1.0 + 0.5 * rng.randn(len(dates))
    dgs10 = 2.0 + 0.5 * rng.randn(len(dates))
    dgs3m = dgs10 - term_spread
    df = pd.DataFrame({
        "date": dates,
        "open": openp.values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "adj close": close.values,
        "volume": vol,
        "vix": vix,
        "DGS10": dgs10,
        "DGS3MO": dgs3m,
    })
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="force offline synthetic data")
    parser.add_argument("--start", default="2010-01-01")
    args = parser.parse_args()

    OUT = Path("data/processed")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.offline:
        print("Using offline synthetic data…")
        market = make_offline_stub(args.start)
    else:
        try:
            print("Fetching market data (SPY, VIX, 10Y, 3M)…")
            market = merge_market(start=args.start)
            if market is None or market.empty:
                print("Remote sources returned empty; switching to offline synthetic data.")
                market = make_offline_stub(args.start)
        except Exception as e:
            print(f"Remote fetch failed: {e}\nSwitching to offline synthetic data.")
            market = make_offline_stub(args.start)

    market.columns = [c.lower() for c in market.columns]
    (OUT / "market_raw.csv").write_text(market.to_csv(index=False))
    feat = add_time_features(market.rename(columns={"adj close": "adj_close"}))
    (OUT / "market.csv").write_text(feat.to_csv(index=False))
    (OUT / "market_head.csv").write_text(feat.head(100).to_csv(index=False))
    print(f"Saved processed to {OUT/'market.csv'} with {len(feat):,} rows\nDone.")

if __name__ == "__main__":
    main()
