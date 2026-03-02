"""
IMDB loader.

Option A (recommended): Hugging Face datasets "imdb"
Option B: local CSV/parquet already standardized

This helper returns a standardized DataFrame with:
user_id (synthetic), item_id (synthetic), review_text, rating (binary label), timestamp, source, sentiment_label (0/1)
"""

from __future__ import annotations
import pandas as pd

def from_hf(split: str = "train", limit: int = 0) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("imdb", split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    df = ds.to_pandas()
    # label: 0 neg, 1 pos
    out = pd.DataFrame({
        "user_id": [f"imdb_u_{i}" for i in range(len(df))],
        "item_id": [f"imdb_item_{i}" for i in range(len(df))],
        "review_text": df["text"].astype(str),
        "rating": None,
        "timestamp": None,
        "source": "imdb",
        "sentiment_label": df["label"].astype(int)
    })
    return out
