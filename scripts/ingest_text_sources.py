import argparse, sys, json, glob
import pandas as pd
from pathlib import Path
from src.nlp.clean_text import clean_text

DATE_CANDIDATES = ["date","datetime","published","published_at","time","timestamp"]
TEXT_CANDIDATES = ["headline","title","text","content","body"]

def _read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".jsonl",".jsonl.txt"]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if ext == ".json":
        obj = json.loads(path.read_text())
        return pd.DataFrame(obj if isinstance(obj, list) else [obj])
    raise ValueError(f"Unsupported file type: {ext} ({path})")

def _auto_cols(df: pd.DataFrame, date_col: str|None, text_col: str|None):
    mapping = {c.lower(): c for c in df.columns}
    dcol = date_col or next((mapping[c] for c in DATE_CANDIDATES if c in mapping), None)
    tcol = text_col or next((mapping[c] for c in TEXT_CANDIDATES if c in mapping), None)
    if dcol is None or tcol is None:
        raise ValueError(f"Could not infer date/text columns. Columns: {list(df.columns)}")
    return dcol, tcol

def main():
    ap = argparse.ArgumentParser(description="Merge CSV/JSON/JSONL into data/raw/headlines.csv (date,headline).")
    ap.add_argument("inputs", nargs="+", help="Input files or globs")
    ap.add_argument("--out", default="data/raw/headlines.csv")
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-per-day", type=int, default=50)
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend([Path(p) for p in glob.glob(pat)])
    if not paths:
        print("No inputs matched.", file=sys.stderr); sys.exit(1)

    rows = []
    for p in paths:
        df = _read_any(p)
        dcol, tcol = _auto_cols(df, args.date_col, args.text_col)
        tmp = df[[dcol, tcol]].rename(columns={dcol:"date", tcol:"headline"}).copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.normalize()
        tmp["headline"] = tmp["headline"].map(clean_text)
        tmp = tmp.dropna(subset=["date"])
        tmp = tmp[tmp["headline"].str.len() >= args.min_chars]
        rows.append(tmp)

    data = pd.concat(rows, ignore_index=True)
    data = data.drop_duplicates(subset=["date","headline"])  # exact dedupe
    data = data.sort_values("date").groupby("date").head(args.max_per_day).reset_index(drop=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)

    cov = data.groupby("date").size().rename("rows").reset_index()
    print(f"Wrote {len(data)} rows to {out}. Unique days: {len(cov)}")
    print("Last few days coverage:"); print(cov.tail(10))

if __name__ == "__main__":
    main()
