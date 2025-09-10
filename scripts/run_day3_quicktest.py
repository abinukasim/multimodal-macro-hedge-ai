
from pathlib import Path
import pandas as pd

def show(path):
    p = Path(path)
    if p.exists():
        df = pd.read_parquet(p)
        print(f"✅ {p.name}: shape={df.shape}")
        print(df.tail())
    else:
        print(f"❌ Missing {p}")

show("data/processed/wf_time_only.parquet")
show("data/processed/wf_fused.parquet")
show("data/processed/curve_time_only.parquet")
show("data/processed/curve_fused.parquet")
