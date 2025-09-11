import pandas as pd
from pathlib import Path
p = Path("data/raw/headlines.csv")
if not p.exists():
    print("Missing data/raw/headlines.csv. Run ingest_text_sources.py first.")
else:
    df = pd.read_csv(p, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = df.groupby("date").size().rename("count").reset_index()
    print("Rows:", len(df), "Unique days:", len(daily))
    print(daily.tail(20))
