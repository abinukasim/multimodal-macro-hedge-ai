
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Macro Multimodal AI", layout="wide")
st.title("📈 Macro Multimodal AI — Dashboard")

@st.cache_data
def load_artifacts():
    p = Path("data/processed")
    data = {}
    for name in ["wf_time_only","wf_fused","curve_time_only","curve_fused"]:
        f = p / f"{name}.parquet"
        if f.exists():
            data[name] = pd.read_parquet(f)
    return data

data = load_artifacts()

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Equity Curves (Net of Costs)")
    if "curve_time_only" in data and "curve_fused" in data:
        curves = data["curve_time_only"][["date","equity"]].rename(columns={"equity":"Time-only"}).merge(
                 data["curve_fused"][["date","equity"]].rename(columns={"equity":"Fused"}),
                 on="date", how="outer").sort_values("date")
        st.line_chart(curves.set_index("date"))
    elif "curve_time_only" in data:
        st.line_chart(data["curve_time_only"].set_index("date")[["equity"]])
    else:
        st.info("Run the training first to generate curves.")

with col2:
    st.subheader("Key Metrics")
    rows = []
    for k, label in [("curve_time_only","Time-only"), ("curve_fused","Fused")]:
        if k in data:
            stats = data[k].attrs.get("stats", {})
            rows.append({
                "Run": label,
                "Net Return": stats.get("Total Return (net)"),
                "CAGR": stats.get("CAGR"),
                "Sharpe": stats.get("Sharpe (net)"),
                "Max DD": stats.get("Max Drawdown"),
                "Trades": stats.get("Trades"),
                "Hit-Rate": stats.get("Hit-Rate"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("No stats found yet.")

st.divider()
st.subheader("Latest Signals")
if "wf_fused" in data:
    st.dataframe(data["wf_fused"].tail(10))
elif "wf_time_only" in data:
    st.dataframe(data["wf_time_only"].tail(10))
else:
    st.write("Run the training to see signals.")
