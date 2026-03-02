from __future__ import annotations
from pathlib import Path
from typing import Optional, Iterable, List
import pandas as pd
from utils.schema import ensure_schema, map_rating_to_sentiment
from preprocessing.text_cleaner import clean_text

def postprocess(df: pd.DataFrame, *, source: str, derive_sentiment_from_rating: bool = True) -> pd.DataFrame:
    # text clean
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("").astype(str).map(clean_text)
    df = ensure_schema(df, source=source)
    if derive_sentiment_from_rating and df["sentiment_label"].isna().all():
        df["sentiment_label"] = map_rating_to_sentiment(df["rating"])
    return df
