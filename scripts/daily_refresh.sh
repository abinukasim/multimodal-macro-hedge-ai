#!/usr/bin/env bash
set -euo pipefail

# Resolve project root even if called from cron
DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY="$DIR/.venv/bin/python"

cd "$DIR"

# 1) fetch fresh official text (pulls first paragraph from pages)
"$PY" -m scripts.fetch_official_text --since 2018-01-01 --outdir mytexts

# 2) merge + features
"$PY" -m scripts.ingest_text_sources mytexts/*.csv --out data/raw/headlines.csv
"$PY" -m scripts.build_text_features --input data/raw/headlines.csv
"$PY" -m scripts.build_fusion_dataset

# 3) retrain (biweekly refits were more robust in your tests)
"$PY" -m scripts.train_baseline --min-date 2018-01-01 --start-idx 252 --step 10 --cost-bps 1

# 4) anchored calibration + prob sizing
"$PY" -m scripts.calibrate_probs --cut 2023-01-01 --sizing prob --prob-scale 0.06

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') • Daily refresh complete."
