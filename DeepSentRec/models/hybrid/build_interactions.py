#!/usr/bin/env python
"""
Module 3 — HybridCF-SBERT: Build user-item interaction data.

Expected input: standardized parquet/csv (Module 1) optionally enriched with sentiment (Module 2).
Required columns:
- user_id
- item_id
Optional:
- rating (explicit)
- timestamp
- click / purchase (if available)

Outputs:
- interactions.parquet: columns [user_id, item_id, rating, weight]
- mappings.npz: user2idx, item2idx arrays for matrix factorization
"""

from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd

def load_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Standardized parquet/csv with user_id,item_id,(rating).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_user_inter", type=int, default=5, help="Filter users with fewer interactions.")
    ap.add_argument("--min_item_inter", type=int, default=5, help="Filter items with fewer interactions.")
    ap.add_argument("--implicit", action="store_true", help="Treat as implicit feedback (rating->1).")
    args = ap.parse_args()

    df = load_table(args.input)
    req = {"user_id","item_id"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "rating" not in df.columns:
        df["rating"] = 1.0
    if args.implicit:
        df["rating"] = 1.0

    # Drop nulls
    df = df.dropna(subset=["user_id","item_id"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    # Basic frequency filtering (iterative 2-pass)
    for _ in range(2):
        ucnt = df["user_id"].value_counts()
        icnt = df["item_id"].value_counts()
        df = df[df["user_id"].isin(ucnt[ucnt>=args.min_user_inter].index)]
        df = df[df["item_id"].isin(icnt[icnt>=args.min_item_inter].index)]

    # Aggregate duplicates
    df = df.groupby(["user_id","item_id"], as_index=False).agg({"rating":"mean"})
    df["weight"] = 1.0

    # Build mappings
    users = df["user_id"].unique()
    items = df["item_id"].unique()
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {it:i for i,it in enumerate(items)}

    os.makedirs(args.out_dir, exist_ok=True)
    out_inter = os.path.join(args.out_dir, "interactions.parquet")
    df.to_parquet(out_inter, index=False)

    np.savez_compressed(
        os.path.join(args.out_dir, "mappings.npz"),
        users=users,
        items=items,
    )

    print(f"Saved: {out_inter} rows={len(df)} users={len(users)} items={len(items)}")

if __name__ == "__main__":
    main()
