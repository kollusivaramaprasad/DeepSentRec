"""
Amazon Reviews loader (UCSD/McAuley style).

Supports:
- json lines (.json) OR gzip compressed json lines (.json.gz)

Outputs a standardized DataFrame with:
user_id, item_id, review_text, rating, timestamp, source
"""

from __future__ import annotations
import gzip, json
from typing import Iterator, Dict, Any, Optional
import pandas as pd

def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.lower().endswith(".gz") else open(path, "r", encoding="utf-8")

def iter_reviews(path: str, limit: int = 0) -> Iterator[Dict[str, Any]]:
    n = 0
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj
            n += 1
            if limit and n >= limit:
                break

def to_standard_df(path: str, limit: int = 0) -> pd.DataFrame:
    rows = []
    for r in iter_reviews(path, limit=limit):
        # common keys: reviewerID, asin, reviewText, overall, unixReviewTime
        rows.append({
            "user_id": str(r.get("reviewerID", "")),
            "item_id": str(r.get("asin", "")),
            "review_text": r.get("reviewText", "") or r.get("summary", "") or "",
            "rating": r.get("overall", None),
            "timestamp": r.get("unixReviewTime", None),
            "source": "amazon"
        })
    df = pd.DataFrame(rows)
    return df
