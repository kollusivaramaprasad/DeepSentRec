from __future__ import annotations
import pandas as pd

def build_interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dense-ish interaction table from unified schema.

    Returns long-form interactions:
    user_id, item_id, rating (or implicit 1.0), timestamp
    """
    out = df[["user_id","item_id","rating","timestamp","source"]].copy()
    # if rating missing, use implicit feedback
    out["implicit"] = out["rating"].isna()
    out.loc[out["implicit"], "rating"] = 1.0
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").astype("float32")
    return out
