import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier  # force XGBoost

TIME_EXCLUDE = {"date","y","open","high","low","close","adj close","adj_close","volume"}

def feature_cols(df: pd.DataFrame, include_text: bool = True) -> list[str]:
    cols = []
    for c in df.columns:
        lc = c.lower()
        if lc in TIME_EXCLUDE:
            continue
        # Speed tip: always ignore huge embedding vectors
        if lc.startswith("emb_"):
            continue
        if (not include_text) and (lc.startswith("finbert_")):
            continue
        cols.append(c)
    return cols

def walk_forward(df: pd.DataFrame, start_idx: int = 252, step: int = 5, include_text: bool = True,
                 xgb_params: dict | None = None) -> pd.DataFrame:
    """Expanding-window walk-forward. Refit every `step` days."""
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=120,          # ↓ fewer trees
            max_depth=4,               # ↓ shallower trees
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            n_jobs=-1,                 # use all CPU cores
            tree_method="hist",        # fast
            eval_metric="logloss",
            random_state=42,
        )

    feats = feature_cols(df, include_text=include_text)
    preds, ys, dates = [], [], []
    n = len(df)

    for i in range(start_idx, n-1, step):
        train = df.iloc[:i]
        test  = df.iloc[i:i+step]  # predict the next `step` days at once

        model = XGBClassifier(**xgb_params)
        model.fit(train[feats], train["y"])
        p = model.predict_proba(test[feats])[:,1]

        preds.extend(p.tolist())
        ys.extend(test["y"].tolist())
        dates.extend(test["date"].tolist())

        # progress ping every ~50 refits
        if ((i - start_idx) // step) % 50 == 0:
            print(f"[walk_forward] {i-start_idx:4d}/{n-start_idx} rows processed | feats={len(feats)}")

    out = pd.DataFrame({"date":dates, "y":ys, "p":preds})
    threshold = 0.55
    out["signal"] = (out["p"] > threshold).astype(int)
    try:
        auc = roc_auc_score(out["y"], out["p"])
    except Exception:
        auc = np.nan
    acc = accuracy_score(out["y"], out["signal"]) if len(out) else np.nan
    pred = out["signal"] if len(out) else pd.Series(dtype=int)
    tp = int(((pred == 1) & (out["y"] == 1)).sum()) if len(out) else 0
    tn = int(((pred == 0) & (out["y"] == 0)).sum()) if len(out) else 0
    fp = int(((pred == 1) & (out["y"] == 0)).sum()) if len(out) else 0
    fn = int(((pred == 0) & (out["y"] == 1)).sum()) if len(out) else 0
    out.attrs["metrics"] = {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "Model": "XGBoost",
        "Threshold": float(threshold),
        "P Mean": float(out["p"].mean()) if len(out) else float("nan"),
        "P Std": float(out["p"].std(ddof=1)) if len(out) > 1 else float("nan"),
        "P01": float(out["p"].quantile(0.01)) if len(out) else float("nan"),
        "P05": float(out["p"].quantile(0.05)) if len(out) else float("nan"),
        "P50": float(out["p"].quantile(0.50)) if len(out) else float("nan"),
        "P95": float(out["p"].quantile(0.95)) if len(out) else float("nan"),
        "P99": float(out["p"].quantile(0.99)) if len(out) else float("nan"),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }
    return out
