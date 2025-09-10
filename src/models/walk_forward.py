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
    out.attrs["metrics"] = {"AUC": float(auc), "Accuracy": float(acc), "Model": "XGBoost"}
    return out