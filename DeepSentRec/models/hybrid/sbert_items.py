#!/usr/bin/env python
"""
Module 3 — SBERT content embeddings for items.

Given a standardized dataset with at least:
- item_id
- review_text

This script produces:
- item_embeddings.npy: [n_items, dim]
- item_ids.npy: parallel array

Strategy:
- For each item, concatenate up to N reviews (or sample) into a single text,
  then encode with sentence-transformers.

Note: For huge datasets, run with --limit_items or pre-aggregate offline.
"""

from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_reviews_per_item", type=int, default=3)
    ap.add_argument("--limit_items", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    df = load_table(args.input)
    if "item_id" not in df.columns or "review_text" not in df.columns:
        raise ValueError("Input must contain item_id and review_text")

    df = df.dropna(subset=["item_id","review_text"])
    df["item_id"] = df["item_id"].astype(str)

    # aggregate
    grouped = df.groupby("item_id")["review_text"].apply(list)
    item_ids = grouped.index.to_numpy()

    if args.limit_items and args.limit_items > 0:
        item_ids = item_ids[:args.limit_items]

    texts = []
    for it in item_ids:
        reviews = grouped.loc[it][:args.max_reviews_per_item]
        texts.append(" ".join([str(r) for r in reviews]))

    st = SentenceTransformer(args.model)
    emb = st.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "item_ids.npy"), item_ids)
    np.save(os.path.join(args.out_dir, "item_embeddings.npy"), emb)
    print(f"Saved item embeddings: {emb.shape} to {args.out_dir}")

if __name__ == "__main__":
    main()
