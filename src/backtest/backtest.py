
import pandas as pd
import numpy as np

def _max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = (equity / cummax) - 1.0
    return float(dd.min()) if len(dd) else float("nan")

def _ann_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean() * periods_per_year
    sd = r.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / sd) if sd > 0 else float("nan")

def pnl_curve(signals_df: pd.DataFrame, price_df: pd.DataFrame, cost_bps: float = 1.0) -> pd.DataFrame:
    df = signals_df.copy().sort_values("date").reset_index(drop=True)
    px = price_df[["date","close"]].copy().sort_values("date")
    out = df.merge(px, on="date", how="left")
    out["ret1"] = out["close"].pct_change()

    if "pos" in out.columns:
        pos_raw = out["pos"].astype(float).clip(lower=0.0, upper=1.0)
    else:
        pos_raw = out["signal"].astype(float).clip(lower=0.0, upper=1.0)

    out["pos"] = pos_raw.shift(1).fillna(0.0)  # enter next bar

    out["strategy_ret"] = out["pos"] * out["ret1"]
    trades = (out["pos"].fillna(0.0) != out["pos"].shift(1).fillna(0.0)).astype(int)
    cost = trades * (cost_bps / 10000.0)
    out["strategy_ret_net"] = out["strategy_ret"] - cost

    out["equity"] = (1.0 + out["strategy_ret_net"].fillna(0.0)).cumprod()
    out["bh_equity"] = (1.0 + out["ret1"].fillna(0.0)).cumprod()

    years = max((out["date"].iloc[-1] - out["date"].iloc[0]).days / 365.25, 1e-9) if len(out) else 0.0
    total_ret_net = float(out["equity"].iloc[-1] - 1.0) if len(out) else float("nan")
    cagr = float(out["equity"].iloc[-1] ** (1.0 / years) - 1.0) if len(out) else float("nan")
    bh_total = float(out["bh_equity"].iloc[-1] - 1.0) if len(out) else float("nan")
    sharpe = _ann_sharpe(out["strategy_ret_net"])
    maxdd = _max_drawdown(out["equity"])
    turnover = float(trades.mean()) if len(trades) else float("nan")

    if "p" in out.columns and "y" in out.columns:
        hr = float(((out["p"] > 0.5).astype(int) == out["y"]).mean())
    else:
        hr = float("nan")

    out.attrs["stats"] = {
        "Total Return (net)": total_ret_net,
        "CAGR": cagr,
        "BH Return": bh_total,
        "Sharpe (net)": sharpe,
        "Max Drawdown": maxdd,
        "Trades": int(trades.sum()),
        "Avg Turnover": turnover,
        "Hit-Rate": hr,
    }
    return out
