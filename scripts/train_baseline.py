import argparse, json
import pandas as pd
from pathlib import Path
from src.data.build_dataset import load_market
from src.models.walk_forward import walk_forward
from src.backtest.backtest import pnl_curve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="data/processed/fusion_dataset.parquet")
    ap.add_argument("--start-idx", type=int, default=252)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=5)          # refit every N days (weekly default)
    ap.add_argument("--min-date", type=str, default=None)   # e.g., 2018-01-01
    args = ap.parse_args()

    # Load data
    if args.fusion.endswith(".csv"):
        X = pd.read_csv(args.fusion, parse_dates=["date"])
    else:
        X = pd.read_parquet(args.fusion)
    X["date"] = pd.to_datetime(X["date"]).dt.normalize()

    # Optional: restrict to recent window for speed
    if args.min_date:
        X = X[X["date"] >= pd.to_datetime(args.min_date)].reset_index(drop=True)

    # Train / predict
    wf_time  = walk_forward(X, start_idx=args.start_idx, step=args.step, include_text=False)
    wf_fused = walk_forward(X, start_idx=args.start_idx, step=args.step, include_text=True)

    # Backtest
    market = load_market("data/processed/market.csv")[["date","close"]]
    curve_time  = pnl_curve(wf_time, market, cost_bps=args.cost_bps)
    curve_fused = pnl_curve(wf_fused, market, cost_bps=args.cost_bps)

    # Save artifacts
    OUT = Path("data/processed")
    wf_time.to_parquet(OUT / "wf_time_only.parquet", index=False)
    wf_fused.to_parquet(OUT / "wf_fused.parquet", index=False)
    curve_time.to_parquet(OUT / "curve_time_only.parquet", index=False)
    curve_fused.to_parquet(OUT / "curve_fused.parquet", index=False)

    # Print summary
    mt = wf_time.attrs.get("metrics", {})
    mf = wf_fused.attrs.get("metrics", {})
    st = curve_time.attrs.get("stats", {})
    sf = curve_fused.attrs.get("stats", {})
    print("\n=== Metrics ===")
    print("Time-only:", json.dumps(mt, indent=2))
    print("Fused    :", json.dumps(mf, indent=2))
    print("\n=== Backtest (net) ===")
    print("Time-only:", json.dumps(st, indent=2))
    print("Fused    :", json.dumps(sf, indent=2))
    print("\nSaved: wf_*.parquet and curve_*.parquet under data/processed")

if __name__ == "__main__":
    main()