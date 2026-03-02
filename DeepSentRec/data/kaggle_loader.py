"""
Kaggle E-commerce reviews loader (CSV).

Because Kaggle datasets vary, this loader supports column mapping.
Default mappings try common names.

Output standardized columns:
user_id, item_id, review_text, rating, timestamp, source
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class KaggleColumnMap:
    user_id: str = "user_id"
    item_id: str = "product_id"
    review_text: str = "review_text"
    rating: str = "rating"
    timestamp: str = "timestamp"

COMMON_FALLBACKS = {
    "user_id": ["user_id","reviewerID","customer_id","UserId","user"],
    "item_id": ["product_id","asin","business_id","item_id","ProductId","product"],
    "review_text": ["review_text","review","text","ReviewText","content","comment"],
    "rating": ["rating","stars","overall","Score","Rating"],
    "timestamp": ["timestamp","time","date","unixReviewTime","reviewTime"]
}

def _pick_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def to_standard_df(path: str, limit: int = 0, colmap: KaggleColumnMap | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit and limit > 0:
        df = df.head(limit)

    # resolve columns
    ucol = _pick_col(df, COMMON_FALLBACKS["user_id"])
    icol = _pick_col(df, COMMON_FALLBACKS["item_id"])
    tcol = _pick_col(df, COMMON_FALLBACKS["review_text"])
    rcol = _pick_col(df, COMMON_FALLBACKS["rating"])
    dcol = _pick_col(df, COMMON_FALLBACKS["timestamp"])

    out = pd.DataFrame({
        "user_id": df[ucol].astype(str) if ucol else "",
        "item_id": df[icol].astype(str) if icol else "",
        "review_text": df[tcol].astype(str) if tcol else "",
        "rating": df[rcol] if rcol else None,
        "timestamp": df[dcol] if dcol else None,
        "source": "kaggle"
    })
    return out
