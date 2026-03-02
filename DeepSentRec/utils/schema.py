"""
DeepSentRec unified data schema utilities.

We standardize all datasets into the same schema so that:
- SentimentBERT (text + label) can be trained consistently
- HybridCF-SBERT (user-item + text embeddings) can be trained consistently
- RLRanker-PPO can consume interaction logs consistently
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLS = ["user_id", "item_id", "review_text", "rating", "timestamp", "source"]
OPTIONAL_COLS = ["review_id", "sentiment_label", "sentiment_score"]

@dataclass(frozen=True)
class Schema:
    required: Tuple[str, ...] = tuple(REQUIRED_COLS)
    optional: Tuple[str, ...] = tuple(OPTIONAL_COLS)

SCHEMA = Schema()

def ensure_schema(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """
    Ensure the dataframe contains all required columns with correct dtypes.

    rating: float (NaN allowed)
    timestamp: int64 unix seconds (NaN allowed)
    """
    df = df.copy()

    # Add missing columns
    for c in SCHEMA.required:
        if c not in df.columns:
            df[c] = pd.NA
    for c in SCHEMA.optional:
        if c not in df.columns:
            df[c] = pd.NA

    # Enforce source
    df["source"] = source

    # Basic cleanup
    df["review_text"] = df["review_text"].fillna("").astype(str)

    # rating -> float
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Float64")

    # timestamp -> int64 unix seconds
    ts = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        df["timestamp"] = (ts.view("int64") // 10**9).astype("Int64")
    else:
        # try parse numeric / datetime-like
        parsed = pd.to_datetime(ts, errors="coerce", utc=True)
        numeric = pd.to_numeric(ts, errors="coerce")
        # prefer numeric if already sensible (>= 10^9)
        use_numeric = numeric.notna() & (numeric.astype("float") >= 1e9)
        out = pd.Series(pd.NA, index=df.index, dtype="Int64")
        out.loc[use_numeric] = numeric.loc[use_numeric].astype("int64")
        out.loc[~use_numeric] = (parsed.loc[~use_numeric].view("int64") // 10**9).astype("Int64")
        df["timestamp"] = out

    # IDs as strings
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    return df[list(SCHEMA.required) + list(SCHEMA.optional)]

def map_rating_to_sentiment(rating: pd.Series) -> pd.Series:
    """
    Map star ratings to {negative, neutral, positive}. Assumes 1–5 scale.
    """
    r = pd.to_numeric(rating, errors="coerce")
    out = pd.Series(pd.NA, index=rating.index, dtype="string")
    out[(r <= 2)] = "negative"
    out[(r == 3)] = "neutral"
    out[(r >= 4)] = "positive"
    return out
