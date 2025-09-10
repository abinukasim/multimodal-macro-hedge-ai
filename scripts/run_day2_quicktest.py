
import pandas as pd
from pathlib import Path

p1 = Path("data/processed/text_features.parquet")
p2 = Path("data/processed/fusion_dataset.parquet")

if not p1.exists():
    print("Text features not found. Run:")
    print("  python -m scripts.build_text_features --input data/raw/headlines.csv")
else:
    tf = pd.read_parquet(p1)
    print("✅ text_features.parquet:", tf.shape, "cols:", list(tf.columns)[:8], "...")

if not p2.exists():
    print("Fusion dataset not found. Run:")
    print("  python -m scripts.build_fusion_dataset")
else:
    X = pd.read_parquet(p2)
    print("✅ fusion_dataset.parquet:", X.shape)
    print(X[["date","close","y","finbert_neg","finbert_neu","finbert_pos"]].tail())
