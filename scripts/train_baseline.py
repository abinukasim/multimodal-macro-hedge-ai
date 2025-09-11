
import argparse, json
import pandas as pd
from pathlib import Path
from src.data.build_dataset import load_market
from src.models.walk_forward import walk_forward
from src.backtest.backtest import pnl_curve

def make_positions(df: pd.DataFrame, sizing: str, threshold: float, band: float, prob_scale: float):
    out = df.copy()
    if sizing == "binary":
        thr = threshold + band
        out["pos"] = (out["p"] >= thr).astype(float)
    elif sizing == "prob":
        z = (out["p"] - 0.5) / max(prob_scale, 1e-6)
        out["pos"] = z.clip(lower=0.0, upper=1.0)
    else:
        raise ValueError("sizing must be 'binary' or 'prob'")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", default="data/processed/fusion_dataset.parquet")
    ap.add_argument("--start-idx", type=int, default=252)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-date", type=str, default=None)
    ap.add_argument("--sizing", choices=["binary","prob"], default="binary")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--band", type=float, default=0.00)
    ap.add_argument("--prob-scale", type=float, default=0.10)
    args = ap.parse_args()

    if args.fusion.endswith(".csv"):
        X = pd.read_csv(args.fusion, parse_dates=["date"])
    else:
        X = pd.read_parquet(args.fusion)
    X["date"] = pd.to_datetime(X["date"]).dt.normalize()

    if args.min_date:
        X = X[X["date"] >= pd.to_datetime(args.min_date)].reset_index(drop=True)

    wf_time  = walk_forward(X, start_idx=args.start_idx, step=args.step, include_text=False)
    wf_fused = walk_forward(X, start_idx=args.start_idx, step=args.step, include_text=True)

    wf_time  = make_positions(wf_time,  args.sizing, args.threshold, args.band, args.prob_scale)
    wf_fused = make_positions(wf_fused, args.sizing, args.threshold, args.band, args.prob_scale)

    market = load_market("data/processed/market.csv")[["date","close"]]
    curve_time  = pnl_curve(wf_time,  market, cost_bps=args.cost_bps)
    curve_fused = pnl_curve(wf_fused, market, cost_bps=args.cost_bps)

    OUT = Path("data/processed")
    wf_time.to_parquet(OUT / "wf_time_only.parquet", index=False)
    wf_fused.to_parquet(OUT / "wf_fused.parquet", index=False)
    curve_time.to_parquet(OUT / "curve_time_only.parquet", index=False)
    curve_fused.to_parquet(OUT / "curve_fused.parquet", index=False)

    print("\n=== Metrics (walk-forward) ===")
    print("Time-only:", json.dumps(wf_time.attrs.get("metrics", {}), indent=2))
    print("Fused    :", json.dumps(wf_fused.attrs.get("metrics", {}), indent=2))
    print("\n=== Backtest (net) ===")
    print("Time-only:", json.dumps(curve_time.attrs.get("stats", {}), indent=2))
    print("Fused    :", json.dumps(curve_fused.attrs.get("stats", {}), indent=2))
    print("\nSaved: wf_*.parquet and curve_*.parquet under data/processed")

if __name__ == "__main__":
    main()
