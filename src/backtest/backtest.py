
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

def _trade_returns(strategy_ret: pd.Series, pos: pd.Series) -> np.ndarray:
    """Return per-trade compounded returns for long/flat regimes."""
    in_pos = pos.fillna(0.0) > 0.0
    if not len(in_pos):
        return np.array([], dtype=float)
    starts = (~in_pos.shift(1, fill_value=False) & in_pos)
    ends = (in_pos & ~in_pos.shift(-1, fill_value=False))
    start_idx = np.flatnonzero(starts.to_numpy())
    end_idx = np.flatnonzero(ends.to_numpy())
    vals = strategy_ret.fillna(0.0).to_numpy()
    rets = []
    for s, e in zip(start_idx, end_idx):
        rets.append(float(np.prod(1.0 + vals[s:e+1]) - 1.0))
    return np.asarray(rets, dtype=float)

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
    turnover = out["pos"].fillna(0.0).sub(out["pos"].shift(1).fillna(0.0)).abs()
    # Cost is applied to traded notional (turnover), once per rebalance.
    cost = turnover * (cost_bps / 10000.0)
    out["strategy_ret_net"] = out["strategy_ret"] - cost

    out["turnover"] = turnover
    out["cost"] = cost
    out["equity"] = (1.0 + out["strategy_ret_net"].fillna(0.0)).cumprod()
    out["equity_gross"] = (1.0 + out["strategy_ret"].fillna(0.0)).cumprod()
    out["bh_equity"] = (1.0 + out["ret1"].fillna(0.0)).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    out["rolling_drawdown_63"] = out["equity"] / out["equity"].rolling(63, min_periods=1).max() - 1.0
    roll_mu = out["strategy_ret_net"].rolling(63).mean() * 252
    roll_sd = out["strategy_ret_net"].rolling(63).std(ddof=1) * np.sqrt(252)
    out["rolling_sharpe_63"] = roll_mu / roll_sd.replace(0.0, np.nan)

    years = max((out["date"].iloc[-1] - out["date"].iloc[0]).days / 365.25, 1e-9) if len(out) else 0.0
    total_ret_net = float(out["equity"].iloc[-1] - 1.0) if len(out) else float("nan")
    total_ret_gross = float(out["equity_gross"].iloc[-1] - 1.0) if len(out) else float("nan")
    cagr = float(out["equity"].iloc[-1] ** (1.0 / years) - 1.0) if len(out) else float("nan")
    cagr_gross = float(out["equity_gross"].iloc[-1] ** (1.0 / years) - 1.0) if len(out) else float("nan")
    bh_total = float(out["bh_equity"].iloc[-1] - 1.0) if len(out) else float("nan")
    sharpe = _ann_sharpe(out["strategy_ret_net"])
    sharpe_gross = _ann_sharpe(out["strategy_ret"])
    maxdd = _max_drawdown(out["equity"])
    avg_turnover = float(turnover.mean()) if len(turnover) else float("nan")
    total_turnover = float(turnover.sum()) if len(turnover) else float("nan")
    trade_days = int((turnover > 0).sum()) if len(turnover) else 0
    long_days = int((out["pos"] > 0).sum()) if len(out) else 0
    flat_days = int((out["pos"] == 0).sum()) if len(out) else 0
    entries = int(((out["pos"] > 0) & (out["pos"].shift(1).fillna(0.0) == 0.0)).sum()) if len(out) else 0
    trade_rets = _trade_returns(out["strategy_ret_net"], out["pos"])
    avg_trade_ret = float(np.nanmean(trade_rets)) if len(trade_rets) else float("nan")

    if "y" in out.columns and (("p_cal" in out.columns) or ("p" in out.columns)):
        prob_col = "p_cal" if "p_cal" in out.columns else "p"
        pred = (out[prob_col] > 0.5).astype(int)
        hr = float((pred == out["y"]).mean())
        tp = int(((pred == 1) & (out["y"] == 1)).sum())
        tn = int(((pred == 0) & (out["y"] == 0)).sum())
        fp = int(((pred == 1) & (out["y"] == 0)).sum())
        fn = int(((pred == 0) & (out["y"] == 1)).sum())
    else:
        hr = float("nan")
        tp = tn = fp = fn = 0

    out.attrs["stats"] = {
        "Total Return (gross)": total_ret_gross,
        "Total Return (net)": total_ret_net,
        "CAGR (gross)": cagr_gross,
        "CAGR": cagr,
        "BH Return": bh_total,
        "Sharpe (gross)": sharpe_gross,
        "Sharpe (net)": sharpe,
        "Max Drawdown": maxdd,
        "Trades": trade_days,
        "Entries": entries,
        "Total Turnover": total_turnover,
        "Avg Turnover": avg_turnover,
        "Long Days": long_days,
        "Flat Days": flat_days,
        "Avg Trade Return (net)": avg_trade_ret,
        "Hit-Rate": hr,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }
    return out
