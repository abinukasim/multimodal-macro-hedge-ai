
from pathlib import Path
import argparse
from src.data.build_dataset import load_market, load_text_features, build_fusion, save_fusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="data/processed/market.csv")
    ap.add_argument("--text", default="data/processed/text_features.parquet")
    ap.add_argument("--out-prefix", default="data/processed/fusion_dataset")
    args = ap.parse_args()

    m = load_market(args.market)
    t = load_text_features(args.text)
    X = build_fusion(m, t, fill_neutral=True)
    save_fusion(X, args.out_prefix)

    print(f"Fusion dataset saved to {args.out_prefix}.parquet with shape {X.shape}")
    cols = [c for c in X.columns if c.startswith("finbert_")] + [c for c in X.columns if c.startswith("emb_")][:5]
    print("Preview columns:", ["date","close","y"] + cols)
    print(X[["date","close","y"] + cols].tail())

if __name__ == "__main__":
    main()
