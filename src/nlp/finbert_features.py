
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _chunk_by_length(text: str, max_chars: int = 1000):
    if not isinstance(text, str) or not text.strip():
        return []
    t = text.strip()
    return [t[i:i+max_chars] for i in range(0, len(t), max_chars)]

class FinbertFeaturizer:
    def __init__(self, use_embeddings: bool = False):
        self.device = _device()
        self.sa_name = "yiyanghkust/finbert-tone"
        self.sa_tok = AutoTokenizer.from_pretrained(self.sa_name)
        self.sa_model = AutoModelForSequenceClassification.from_pretrained(self.sa_name).to(self.device).eval()
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.emb_name = "ProsusAI/finbert"
            self.emb_tok = AutoTokenizer.from_pretrained(self.emb_name)
            self.emb_model = AutoModel.from_pretrained(self.emb_name).to(self.device).eval()
            self.emb_dim = self.emb_model.config.hidden_size
        else:
            self.emb_model = None
            self.emb_dim = 0

    @torch.no_grad()
    def _sa_probs(self, text: str, max_length: int = 256):
        if not text:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
        inputs = self.sa_tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        logits = self.sa_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        return probs.astype(np.float32)

    @torch.no_grad()
    def _emb_vec(self, text: str, max_length: int = 256):
        if not self.use_embeddings:
            return np.zeros(0, dtype=np.float32)
        if not text:
            return np.zeros(self.emb_dim, dtype=np.float32)
        inputs = self.emb_tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        last = self.emb_model(**inputs).last_hidden_state
        vec = last.mean(dim=1).detach().cpu().numpy()[0]
        return vec.astype(np.float32)

    def featurize_row(self, date, text: str):
        chunks = _chunk_by_length(text, max_chars=1000)
        if not chunks:
            sa = np.array([1/3, 1/3, 1/3], dtype=np.float32)
            emb = np.zeros(self.emb_dim, dtype=np.float32) if self.use_embeddings else np.zeros(0, dtype=np.float32)
        else:
            sa_list, emb_list = [], []
            for c in chunks:
                sa_list.append(self._sa_probs(c))
                if self.use_embeddings:
                    emb_list.append(self._emb_vec(c))
            sa = np.stack(sa_list, axis=0).mean(axis=0) if sa_list else np.array([1/3,1/3,1/3], dtype=np.float32)
            emb = (np.stack(emb_list, axis=0).mean(axis=0) if (self.use_embeddings and emb_list) else
                   (np.zeros(self.emb_dim, dtype=np.float32) if self.use_embeddings else np.zeros(0, dtype=np.float32)))
        out = {
            "date": pd.to_datetime(date).normalize(),
            "finbert_neg": float(sa[0]),
            "finbert_neu": float(sa[1]),
            "finbert_pos": float(sa[2]),
        }
        if self.use_embeddings:
            for i, v in enumerate(emb.tolist()):
                out[f"emb_{i}"] = float(v)
        return out

def build_finbert_features(df_text: pd.DataFrame, date_col: str = "date", text_col: str = "corpus_text", use_embeddings: bool = False) -> pd.DataFrame:
    df_text = df_text.copy()
    df_text[date_col] = pd.to_datetime(df_text[date_col]).dt.normalize()
    agg = df_text.groupby(date_col)[text_col].apply(lambda s: "\n".join([str(x) for x in s if isinstance(x, str)])).reset_index()
    fe = FinbertFeaturizer(use_embeddings=use_embeddings)
    rows = [fe.featurize_row(row[date_col], row[text_col]) for _, row in agg.iterrows()]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
