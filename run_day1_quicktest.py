
from pathlib import Path
import pandas as pd

p = Path("data/processed/market.csv")
if not p.exists():
    print("market.csv not found. Run: python scripts/bootstrap_data.py")
else:
    df = pd.read_csv(p, nrows=5)
    print("✅ market.csv found. Preview:")
    print(df.head())
