
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Macro Multimodal AI", layout="wide")
st.title("📈 Macro Multimodal AI — Starter Dashboard")

st.write("This is a placeholder. After Day 5, this will show signals, equity curve, and ablation results.")

try:
    df = pd.read_csv("data/processed/market_head.csv")
    st.subheader("Market sample (head)")
    st.dataframe(df.head(20))
except Exception as e:
    st.info("Run `python scripts/bootstrap_data.py` to generate data first.")
