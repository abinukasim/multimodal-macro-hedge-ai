
# Day 2 — Text Features (FinBERT) + Fusion Join

## 0) Install extra deps
```bash
source .venv/bin/activate   # if not already active
pip install transformers torch sentencepiece tqdm
```

## 1) Add sample headlines
A small `data/raw/headlines.csv` is included. You can append your own rows (columns: `date,headline`). Dates must be ISO-like (YYYY-MM-DD).

## 2) Build FinBERT text features
Fast (sentiment only):
```bash
python -m scripts.build_text_features --input data/raw/headlines.csv
```

With embeddings (slower, adds ~768 `emb_*` columns):
```bash
python -m scripts.build_text_features --input data/raw/headlines.csv --use-embeddings
```

Output: `data/processed/text_features.parquet` (+ CSV).

## 3) Join with market to build the fusion dataset
```bash
python -m scripts.build_fusion_dataset
```
Output: `data/processed/fusion_dataset.parquet` (+ CSV) with label `y = 1{close_{t+1} > close_t}`.

## 4) Quick test
```bash
python -m scripts.run_day2_quicktest.py
```

## Notes
- The join **does not forward-fill** text. Missing days get neutral sentiment (1/3 each) and zero embeddings, which is a safe baseline.
- FinBERT will download the models on first run and cache them under `~/.cache/huggingface/`.
- On Apple Silicon, the script uses `mps` automatically when available.
