import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Macro Multimodal AI", layout="wide")

st.markdown(
    """
    <style>
      :root {
        --bg: #07101d;
        --panel: #0d1828;
        --panel-alt: #0b1422;
        --line: #253750;
        --text: #e5e7eb;
        --muted: #93a4ba;
        --baseline: #97a7bb;
        --accent: #25c7e8;
        --good: #3cb371;
        --bad: #d46b6b;
      }
      [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 700px at 8% -20%, #1a3259 0%, var(--bg) 48%);
      }
      .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
      }
      h1, h2, h3, h4, p, li, span, label {
        color: var(--text);
      }
      .hero-box {
        background: linear-gradient(170deg, #0f2036 0%, #0b1625 100%);
        border: 1px solid #314867;
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin: 0.2rem 0 0.95rem 0;
      }
      .hero-title {
        font-size: 1.05rem;
        font-weight: 700;
      }
      .hero-sub {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.25rem;
      }
      .kpi-card {
        background: linear-gradient(170deg, var(--panel) 0%, var(--panel-alt) 100%);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 12px 14px 12px 14px;
        margin-bottom: 6px;
      }
      .kpi-card-primary {
        min-height: 156px;
      }
      .kpi-card-secondary {
        min-height: 126px;
      }
      .kpi-card-featured {
        border-color: #2faed0;
        box-shadow: 0 0 0 1px rgba(37, 199, 232, 0.12) inset;
      }
      .kpi-label {
        color: var(--muted);
        font-size: 0.77rem;
        font-weight: 650;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      .kpi-value-primary {
        font-size: 2.05rem;
        font-weight: 760;
        line-height: 1.1;
        margin-top: 0.36rem;
        margin-bottom: 0.42rem;
      }
      .kpi-value-secondary {
        font-size: 1.65rem;
        font-weight: 720;
        line-height: 1.1;
        margin-top: 0.34rem;
        margin-bottom: 0.36rem;
      }
      .delta-positive {
        color: var(--good);
        font-size: 0.9rem;
        font-weight: 620;
      }
      .delta-negative {
        color: var(--bad);
        font-size: 0.9rem;
        font-weight: 620;
      }
      .delta-neutral {
        color: var(--muted);
        font-size: 0.9rem;
      }
      .kpi-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.18rem;
      }
      [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
      }
      .section-gap {
        margin-top: 0.5rem;
      }
      .chart-wrap {
        background: linear-gradient(175deg, #0c1626 0%, #0a1320 100%);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.35rem 0.35rem 0.25rem 0.35rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts():
    p = Path("data/processed")
    data = {}
    for name in ["wf_time_only", "wf_fused", "wf_fused_cal", "curve_time_only", "curve_fused"]:
        f = p / f"{name}.parquet"
        if f.exists():
            data[name] = pd.read_parquet(f)
    return data


def fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.2%}"


def fmt_num(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def delta_text(fused, baseline, percent=False, higher_is_better=True):
    if fused is None or baseline is None or pd.isna(fused) or pd.isna(baseline):
        return "n/a", "neutral"
    diff = float(fused) - float(baseline)
    better = diff >= 0 if higher_is_better else diff <= 0
    direction = "higher" if diff >= 0 else "lower"
    if percent:
        text = f"{abs(diff):.2%} {direction} vs Time-only"
    else:
        text = f"{abs(diff):.2f} {direction} vs Time-only"
    return text, ("positive" if better else "negative")


def drawdown_delta_text(fused_dd, base_dd):
    if fused_dd is None or base_dd is None or pd.isna(fused_dd) or pd.isna(base_dd):
        return "n/a", "neutral"
    improvement = abs(float(base_dd)) - abs(float(fused_dd))
    if improvement > 0:
        return f"{improvement:.2%} lower drawdown vs Time-only", "positive"
    if improvement < 0:
        return f"{abs(improvement):.2%} higher drawdown vs Time-only", "negative"
    return "Same drawdown as Time-only", "neutral"


def trade_delta_text(fused_trades, base_trades):
    if fused_trades is None or base_trades is None or pd.isna(fused_trades) or pd.isna(base_trades):
        return "n/a", "neutral"
    fewer = int(round(float(base_trades) - float(fused_trades)))
    if fewer > 0:
        return f"{fewer} fewer trades vs Time-only", "positive"
    if fewer < 0:
        return f"{abs(fewer)} more trades vs Time-only", "negative"
    return "Same trades as Time-only", "neutral"


def rolling_hit_rate(df: pd.DataFrame, prob_col: str, window: int = 63) -> pd.DataFrame:
    out = df[["date", "y", prob_col]].copy().sort_values("date")
    out["pred"] = (out[prob_col] > 0.5).astype(int)
    out["hit"] = (out["pred"] == out["y"]).astype(float)
    out["rolling_hit_rate"] = out["hit"].rolling(window, min_periods=window // 2).mean()
    return out[["date", "rolling_hit_rate"]]


def drawdown_frame(df: pd.DataFrame, col: str = "drawdown") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns and "equity" in out.columns:
        out[col] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def render_kpi_card(column, title, value, delta, delta_kind="neutral", primary=True, featured=False, note=None):
    cls = "kpi-card kpi-card-primary" if primary else "kpi-card kpi-card-secondary"
    if featured:
        cls = f"{cls} kpi-card-featured"
    delta_cls = f"delta-{delta_kind}"
    value_cls = "kpi-value-primary" if primary else "kpi-value-secondary"
    note_html = f"<div class='kpi-note'>{note}</div>" if note else ""
    column.markdown(
        f"""
        <div class="{cls}">
          <div class="kpi-label">{title}</div>
          <div class="{value_cls}">{value}</div>
          <div class="{delta_cls}">{delta}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_metrics_table(df: pd.DataFrame):
    def highlight_best(col: pd.Series):
        styles = [""] * len(col)
        if col.name == "Trades":
            best_i = col.idxmin()
        elif col.name == "Max DD":
            best_i = col.idxmax()
        elif col.name == "Strategy":
            return styles
        else:
            best_i = col.idxmax()
        for i in range(len(col)):
            if col.index[i] == best_i:
                styles[i] = "background-color: rgba(60,179,113,0.18); color: #9ae6b4; font-weight: 700;"
        return styles

    return (
        df.style
        .format(
            {
                "Net Return": "{:.2%}",
                "Gross Return": "{:.2%}",
                "CAGR": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max DD": "{:.2%}",
                "Hit-Rate": "{:.2%}",
                "Trades": "{:.0f}",
            }
        )
        .apply(highlight_best, axis=0)
        .set_properties(**{"color": "#e5e7eb"})
    )


def apply_chart_style(chart):
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            grid=True,
            gridColor="#1f3048",
            gridOpacity=0.45,
            domainColor="#415772",
            tickColor="#415772",
            labelColor="#cbd5e1",
            titleColor="#cbd5e1",
        )
        .configure_legend(
            labelColor="#dbe4f0",
            titleColor="#dbe4f0",
            symbolStrokeWidth=3,
            orient="top",
            direction="horizontal",
        )
        .configure_title(color="#e2e8f0", fontSize=16, anchor="start")
    )


def equity_chart(curves_df: pd.DataFrame):
    long_df = curves_df.melt("date", var_name="Strategy", value_name="Equity").dropna()
    base = alt.Chart(long_df).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Equity:Q", title="Growth of $1"),
        color=alt.Color(
            "Strategy:N",
            scale=alt.Scale(domain=["Time-only", "Fused / Calibrated"], range=["#95a3b8", "#25c7e8"]),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Strategy:N"),
            alt.Tooltip("Equity:Q", format=".3f"),
        ],
    )
    lines = base.mark_line().encode(
        size=alt.condition(alt.datum.Strategy == "Fused / Calibrated", alt.value(3.9), alt.value(2.2)),
        opacity=alt.condition(alt.datum.Strategy == "Fused / Calibrated", alt.value(1.0), alt.value(0.72)),
    )
    chart = lines.properties(height=430, title="Equity Curve (Net of Costs)").interactive()
    return apply_chart_style(chart)


def drawdown_chart(dd_df: pd.DataFrame):
    long_df = dd_df.melt("date", var_name="Strategy", value_name="Drawdown").dropna()
    base = alt.Chart(long_df).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "Strategy:N",
            scale=alt.Scale(domain=["Time-only", "Fused / Calibrated"], range=["#95a3b8", "#25c7e8"]),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Strategy:N"),
            alt.Tooltip("Drawdown:Q", format=".2%"),
        ],
    )
    lines = base.mark_line().encode(
        size=alt.condition(alt.datum.Strategy == "Fused / Calibrated", alt.value(3.5), alt.value(2.0)),
        opacity=alt.condition(alt.datum.Strategy == "Fused / Calibrated", alt.value(1.0), alt.value(0.72)),
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#4f637f", strokeDash=[4, 4]).encode(y="y:Q")
    chart = (zero + lines).properties(height=335, title="Drawdown Through Time").interactive()
    return apply_chart_style(chart)


def rolling_acc_chart(df: pd.DataFrame):
    long_df = df.melt("date", var_name="Strategy", value_name="Rolling Hit Rate").dropna()
    base = alt.Chart(long_df).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Rolling Hit Rate:Q", title="Rolling Hit Rate", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "Strategy:N",
            scale=alt.Scale(
                domain=["Time-only", "Fused / Calibrated", "Fused"],
                range=["#95a3b8", "#25c7e8", "#4fd1c5"],
            ),
            legend=alt.Legend(title=None),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Strategy:N"),
            alt.Tooltip("Rolling Hit Rate:Q", format=".2%"),
        ],
    )
    lines = base.mark_line().encode(
        size=alt.condition(
            (alt.datum.Strategy == "Fused / Calibrated") | (alt.datum.Strategy == "Fused"),
            alt.value(3.1),
            alt.value(2.0),
        ),
        opacity=alt.condition(
            (alt.datum.Strategy == "Fused / Calibrated") | (alt.datum.Strategy == "Fused"),
            alt.value(1.0),
            alt.value(0.72),
        ),
    )
    mid = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color="#4f637f", strokeDash=[4, 4]).encode(y="y:Q")
    chart = (mid + lines).properties(height=320, title="Rolling Directional Accuracy (63D)").interactive()
    return apply_chart_style(chart)


data = load_artifacts()
time_stats = data.get("curve_time_only", pd.DataFrame()).attrs.get("stats", {})
fused_stats = data.get("curve_fused", pd.DataFrame()).attrs.get("stats", {})

signal_df = None
for key in ["wf_fused_cal", "wf_fused", "wf_time_only"]:
    if key in data:
        signal_df = data[key]
        break

latest_date = None
latest_pos = None
if signal_df is not None and not signal_df.empty:
    signal_df = signal_df.sort_values("date")
    latest_date = signal_df["date"].iloc[-1]
    latest_pos = float(signal_df["pos"].iloc[-1]) if "pos" in signal_df.columns else float(signal_df["signal"].iloc[-1])

st.title("Macro Multimodal Hedge AI")
st.caption("Institutional research dashboard: time-only baseline vs fused/calibrated strategy")
st.markdown(
    """
    <div class="hero-box">
      <div class="hero-title">Executive Summary</div>
      <div class="hero-sub">
        Out-of-sample walk-forward framework combining market features with macro-text sentiment.
        Fused/calibrated results are highlighted as the primary strategy for presentation and risk-aware comparison.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.header("Project Overview")
st.markdown(
    """
    - **Time-only baseline:** market returns, volatility, and rates/term-spread features.
    - **Fused / calibrated strategy:** adds macro-text sentiment and probability calibration for sizing.
    - **Evaluation:** transaction-cost-aware walk-forward backtest, compared side by side.
    """
)

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.header("Headline KPIs")

ret_delta_txt, ret_delta_kind = delta_text(
    fused_stats.get("Total Return (net)"), time_stats.get("Total Return (net)"), percent=True, higher_is_better=True
)
sharpe_delta_txt, sharpe_delta_kind = delta_text(
    fused_stats.get("Sharpe (net)"), time_stats.get("Sharpe (net)"), percent=False, higher_is_better=True
)
dd_delta_txt, dd_delta_kind = drawdown_delta_text(fused_stats.get("Max Drawdown"), time_stats.get("Max Drawdown"))
trade_delta_txt, trade_delta_kind = trade_delta_text(fused_stats.get("Trades"), time_stats.get("Trades"))
hit_delta_txt, hit_delta_kind = delta_text(
    fused_stats.get("Hit-Rate"), time_stats.get("Hit-Rate"), percent=True, higher_is_better=True
)

primary = st.columns(3)
render_kpi_card(
    primary[0],
    "Fused / Calibrated Net Return",
    fmt_pct(fused_stats.get("Total Return (net)")),
    ret_delta_txt,
    ret_delta_kind,
    primary=True,
    featured=True,
)
render_kpi_card(
    primary[1],
    "Fused / Calibrated Sharpe",
    fmt_num(fused_stats.get("Sharpe (net)"), 2),
    sharpe_delta_txt,
    sharpe_delta_kind,
    primary=True,
)
render_kpi_card(
    primary[2],
    "Fused / Calibrated Max Drawdown",
    fmt_pct(fused_stats.get("Max Drawdown")),
    dd_delta_txt,
    dd_delta_kind,
    primary=True,
)

secondary = st.columns(3)
render_kpi_card(
    secondary[0],
    "Fused / Calibrated Trades",
    "n/a" if fused_stats.get("Trades") is None else f"{int(fused_stats.get('Trades'))}",
    trade_delta_txt,
    trade_delta_kind,
    primary=False,
)
render_kpi_card(
    secondary[1],
    "Fused / Calibrated Hit Rate",
    fmt_pct(fused_stats.get("Hit-Rate")),
    hit_delta_txt,
    hit_delta_kind,
    primary=False,
)
render_kpi_card(
    secondary[2],
    "Latest Position",
    "n/a" if latest_pos is None else f"{latest_pos:.2f}",
    "Latest live signal" if latest_date is None else f"as of {pd.to_datetime(latest_date).date()}",
    "neutral",
    primary=False,
)

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.header("Performance")
if "curve_time_only" in data and "curve_fused" in data:
    curves = (
        data["curve_time_only"][["date", "equity"]]
        .rename(columns={"equity": "Time-only"})
        .merge(
            data["curve_fused"][["date", "equity"]].rename(columns={"equity": "Fused / Calibrated"}),
            on="date",
            how="outer",
        )
        .sort_values("date")
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.altair_chart(equity_chart(curves), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
elif "curve_time_only" in data:
    st.line_chart(data["curve_time_only"].set_index("date")[["equity"]], height=430)
else:
    st.info("Run training first to generate performance curves.")

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.header("Drawdown")
if "curve_time_only" in data and "curve_fused" in data:
    c_time = drawdown_frame(data["curve_time_only"])
    c_fused = drawdown_frame(data["curve_fused"])
    dd = (
        c_time[["date", "drawdown"]]
        .rename(columns={"drawdown": "Time-only"})
        .merge(
            c_fused[["date", "drawdown"]].rename(columns={"drawdown": "Fused / Calibrated"}),
            on="date",
            how="outer",
        )
        .sort_values("date")
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.altair_chart(drawdown_chart(dd), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
elif "curve_time_only" in data:
    c_time = drawdown_frame(data["curve_time_only"])
    st.line_chart(c_time.set_index("date")[["drawdown"]], height=335)
else:
    st.info("No drawdown series found.")

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.header("Rolling Directional Accuracy")
rh_cols = []
if "wf_time_only" in data:
    r1 = rolling_hit_rate(data["wf_time_only"], "p").rename(columns={"rolling_hit_rate": "Time-only"})
    rh_cols.append(r1)
if "wf_fused_cal" in data:
    r2 = rolling_hit_rate(data["wf_fused_cal"], "p_cal").rename(columns={"rolling_hit_rate": "Fused / Calibrated"})
    rh_cols.append(r2)
elif "wf_fused" in data:
    r2 = rolling_hit_rate(data["wf_fused"], "p").rename(columns={"rolling_hit_rate": "Fused"})
    rh_cols.append(r2)

if rh_cols:
    rolling_df = rh_cols[0]
    for nxt in rh_cols[1:]:
        rolling_df = rolling_df.merge(nxt, on="date", how="outer")
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.altair_chart(rolling_acc_chart(rolling_df.sort_values("date")), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Walk-forward outputs not found for rolling accuracy view.")

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.header("Key Metrics")
rows = []
for k, label in [("curve_time_only", "Time-only"), ("curve_fused", "Fused / Calibrated")]:
    if k in data:
        stats = data[k].attrs.get("stats", {})
        rows.append(
            {
                "Strategy": label,
                "Net Return": stats.get("Total Return (net)"),
                "Gross Return": stats.get("Total Return (gross)"),
                "CAGR": stats.get("CAGR"),
                "Sharpe": stats.get("Sharpe (net)"),
                "Max DD": stats.get("Max Drawdown"),
                "Trades": float(stats.get("Trades", np.nan)),
                "Hit-Rate": stats.get("Hit-Rate"),
            }
        )
if rows:
    table = pd.DataFrame(rows)
    st.dataframe(styled_metrics_table(table), use_container_width=True, hide_index=True)
else:
    st.info("No metrics found yet.")

st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
st.header("How It Works")
st.markdown(
    """
    1. **Ingest** market and macro-text data.
    2. **Engineer features** from prices/rates and FinBERT sentiment.
    3. **Train walk-forward** to estimate next-day direction probability.
    4. **Size positions** with threshold or calibrated probabilities.
    5. **Backtest net of costs** and compare risk-adjusted outcomes.
    """
)

takeaway = [
    "Fusion helps suppress marginal signals, often reducing unnecessary trading.",
    "Calibration stabilizes probability scaling and supports smoother exposure control.",
    "In this run, risk behavior (drawdown and turnover) improved more reliably than raw hit-rate."
]
if time_stats and fused_stats:
    dd_improvement = abs(float(time_stats.get("Max Drawdown", np.nan))) - abs(float(fused_stats.get("Max Drawdown", np.nan)))
    trade_change = float(time_stats.get("Trades", np.nan)) - float(fused_stats.get("Trades", np.nan))
    if not np.isnan(dd_improvement):
        takeaway[0] = f"Fused / calibrated lowered max drawdown by about {dd_improvement:.2%} versus time-only in this run."
    if not np.isnan(trade_change):
        if trade_change > 0:
            takeaway[1] = f"Fused / calibrated executed about {int(trade_change)} fewer trades versus time-only."
        elif trade_change < 0:
            takeaway[1] = f"Fused / calibrated executed about {int(abs(trade_change))} more trades versus time-only."
        else:
            takeaway[1] = "Trade count was approximately the same as time-only."

st.header("Key Takeaways")
for t in takeaway:
    st.markdown(f"- {t}")

st.header("Latest Signals")
if signal_df is not None and not signal_df.empty:
    show_cols = [c for c in ["date", "y", "p", "p_cal", "signal", "pos"] if c in signal_df.columns]
    st.dataframe(signal_df[show_cols].sort_values("date").tail(12), use_container_width=True, hide_index=True)
else:
    st.info("Run the training pipeline to generate latest signals.")
