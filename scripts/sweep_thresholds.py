import argparse, itertools, json
import pandas as pd
from pathlib import Path
from src.backtest.backtest import pnl_curve

def make_positions(df: pd.DataFrame, sizing: str, threshold: float, band: float, prob_scale: float):
    out = df.copy()
    if sizing == "binary":
        thr = threshold + band
        out["pos"] = (out["p"] >= thr).astype(float)
    else:
        z = (out["p"] - 0.5) / max(prob_scale, 1e-6)
        out["pos"] = z.clip(0, 1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", default="data/processed/wf_fused.parquet")
    ap.add_argument("--market", default="data/processed/market.csv")
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--sizing", choices=["binary","prob"], default="binary")
    ap.add_argument("--thresholds", default="0.55,0.57,0.60")
    ap.add_argument("--bands", default="0.00,0.02,0.05")
    ap.add_argument("--prob-scales", default="0.08,0.10,0.15")
    args = ap.parse_args()

    wf = pd.read_parquet(args.wf)
    market = pd.read_csv(args.market, parse_dates=["date"])

    results = []
    if args.sizing == "binary":
        for T, B in itertools.product(map(float, args.thresholds.split(",")),
                                      map(float, args.bands.split(","))):
            df = make_positions(wf, "binary", T, B, 0.1)
            curve = pnl_curve(df, market, cost_bps=args.cost_bps)
            stats = curve.attrs.get("stats", {}).copy()
            stats.update({"sizing":"binary","threshold":T,"band":B})
            results.append(stats)
    else:
        for PS in map(float, args.prob_scales.split(",")):
            df = make_positions(wf, "prob", 0.5, 0.0, PS)
            curve = pnl_curve(df, market, cost_bps=args.cost_bps)
            stats = curve.attrs.get("stats", {}).copy()
            stats.update({"sizing":"prob","prob_scale":PS})
            results.append(stats)

    out = pd.DataFrame(results)
    out = out.sort_values(["Sharpe (net)","Total Return (net)"], ascending=False)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/processed/sweep_results.csv", index=False)
    print(out.head(10).to_string(index=False))
    print("\nSaved results -> data/processed/sweep_results.csv")
if __name__ == "__main__":
    main()
