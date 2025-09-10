
import argparse
import pandas as pd
from pathlib import Path
from src.nlp.finbert_features import build_finbert_features

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/headlines.csv", help="CSV with columns: date,headline")
    p.add_argument("--use-embeddings", action="store_true", help="Also compute FinBERT pooled embeddings (slower)")
    args = p.parse_args()

    IN = Path(args.input)
    OUT = Path("data/processed")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN, parse_dates=["date"])
    df = df.rename(columns={"headline": "corpus_text"})
    feats = build_finbert_features(df, date_col="date", text_col="corpus_text", use_embeddings=args.use_embeddings)

    feats.to_parquet(OUT / "text_features.parquet", index=False)
    feats.to_csv(OUT / "text_features.csv", index=False)
    print(f"Saved {len(feats)} daily text feature rows to {OUT/'text_features.parquet'}")

if __name__ == "__main__":
    main()
