"""
Yelp Open Dataset loader (review.json).

Input: JSON Lines file with keys:
user_id, business_id, text, stars, date

Output standardized columns:
user_id, item_id (business_id), review_text, rating, timestamp, source
"""

from __future__ import annotations
import json
from typing import Iterator, Dict, Any
import pandas as pd

def iter_reviews(path: str, limit: int = 0) -> Iterator[Dict[str, Any]]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
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
        rows.append({
            "user_id": str(r.get("user_id", "")),
            "item_id": str(r.get("business_id", "")),
            "review_text": r.get("text", "") or "",
            "rating": r.get("stars", None),
            "timestamp": r.get("date", None),
            "source": "yelp"
        })
    return pd.DataFrame(rows)
