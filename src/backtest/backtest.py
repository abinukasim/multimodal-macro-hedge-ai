
import pandas as pd
import numpy as np

def pnl_curve(signals_df: pd.DataFrame, price_df: pd.DataFrame, cost_bps: float = 1.0) -> pd.DataFrame:
    df = signals_df.copy()
    px = price_df[["date","close"]].copy()
    out = df.merge(px, on="date", how="left").sort_values("date").reset_index(drop=True)
    out["ret1"] = out["close"].pct_change()
    out["pos"] = out["signal"].shift(1).fillna(0)  # enter next bar
    out["strategy_ret"] = out["pos"] * out["ret1"]
    trades = (out["pos"] != out["pos"].shift(1)).astype(int).fillna(0)
    cost = trades * (cost_bps / 10000.0)
    out["strategy_ret_net"] = out["strategy_ret"] - cost
    out["equity"] = (1.0 + out["strategy_ret_net"]).cumprod()
    out["bh_equity"] = (1.0 + out["ret1"].fillna(0)).cumprod()
    out.attrs["stats"] = {
        "Total Return (net)": float(out["equity"].iloc[-1] - 1.0) if len(out) else float("nan"),
        "BH Return": float(out["bh_equity"].iloc[-1] - 1.0) if len(out) else float("nan"),
        "Trades": int(trades.sum()),
        "Avg Turnover": float(trades.mean()),
    }
    return out
