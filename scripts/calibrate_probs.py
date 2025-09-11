import argparse
from pathlib import Path
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from src.backtest.backtest import pnl_curve

def build_positions(df, sizing, threshold, band, prob_scale):
    out = df.copy()
    if sizing == "prob":
        out["pos"] = ((out["p_cal"] - 0.5) / max(prob_scale, 1e-6)).clip(0, 1)
        out["signal"] = (out["pos"] > 0).astype(int)
    else:
        thr = threshold + band
        out["signal"] = (out["p_cal"] >= thr).astype(int)
        out["pos"] = out["signal"].astype(float)
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Leakage-safe (anchored split) probability calibration + sizing."
    )
    ap.add_argument("--wf", default="data/processed/wf_fused.parquet")
    ap.add_argument("--market", default="data/processed/market.csv")
    ap.add_argument("--cut", default="2023-01-01", help="Calibrate on dates < CUT; trade on >= CUT")
    ap.add_argument("--sizing", choices=["prob","binary"], default="prob")
    ap.add_argument("--prob-scale", type=float, default=0.06)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--band", type=float, default=0.00)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--out-wf", default="data/processed/wf_fused_cal.parquet")
    ap.add_argument("--out-curve", default="data/processed/curve_fused.parquet")
    args = ap.parse_args()

    wf = pd.read_parquet(args.wf).sort_values("date")
    mkt = pd.read_csv(args.market, parse_dates=["date"])

    cut = pd.to_datetime(args.cut)
    cal = wf[wf["date"] < cut]
    live = wf[wf["date"] >= cut].copy()

    if len(cal) < 50:
        raise SystemExit(f"Not enough calibration data before {args.cut} (got {len(cal)})")

    ir = IsotonicRegression(out_of_bounds="clip").fit(cal["p"], cal["y"])
    live["p_cal"] = ir.transform(live["p"])

    # keep original p for reference; use p_cal for sizing
    live = build_positions(
        live, args.sizing, args.threshold, args.band, args.prob_scale
    )

    # Save calibrated WF (so Streamlit can show latest signals if you want)
    Path(args.out_wf).parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["date","y","p","p_cal","signal","pos"] if c in live.columns]
    live[cols].to_parquet(args.out_wf, index=False)

    # Build and save curve for dashboard
    curve = pnl_curve(live, mkt, cost_bps=args.cost_bps)
    curve.to_parquet(args.out_curve, index=False)
    print("\n=== Calibrated Stats ===")
    print(curve.attrs.get("stats", {}))

if __name__ == "__main__":
    main()
