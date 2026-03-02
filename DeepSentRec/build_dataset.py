#!/usr/bin/env python
"""
Module 1 — Build standardized datasets (parquet/csv).

Supported datasets:
- amazon: UCSD/McAuley JSON lines (.json or .json.gz)
- yelp: Yelp review.json JSON Lines
- imdb_hf: HuggingFace imdb split
- kaggle: CSV with flexible columns

Produces standardized table with:
user_id, item_id, review_text, rating, timestamp, source (+ optional sentiment_label for imdb)

Optionally applies preprocessing cleaning.
"""

from __future__ import annotations
import argparse, os
import pandas as pd

from DeepSentRec.preprocessing.text_cleaner import clean_text
from DeepSentRec.data.amazon_loader import to_standard_df as amazon_df
from DeepSentRec.data.yelp_loader import to_standard_df as yelp_df
from DeepSentRec.data.kaggle_loader import to_standard_df as kaggle_df
from DeepSentRec.data.imdb_loader import from_hf as imdb_df

def save_df(df: pd.DataFrame, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if out.lower().endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["amazon","yelp","imdb_hf","kaggle"], required=True)
    ap.add_argument("--input", default="", help="Input file path for amazon/yelp/kaggle.")
    ap.add_argument("--split", default="train", help="For imdb_hf: train/test.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.dataset == "amazon":
        if not args.input:
            raise ValueError("--input required for amazon")
        df = amazon_df(args.input, limit=args.limit)
    elif args.dataset == "yelp":
        if not args.input:
            raise ValueError("--input required for yelp")
        df = yelp_df(args.input, limit=args.limit)
    elif args.dataset == "kaggle":
        if not args.input:
            raise ValueError("--input required for kaggle")
        df = kaggle_df(args.input, limit=args.limit)
    else:
        df = imdb_df(split=args.split, limit=args.limit)

    if args.clean:
        df["review_text"] = df["review_text"].map(clean_text)

    save_df(df, args.out)
    print(f"Saved: {args.out} rows={len(df)} cols={list(df.columns)}")

if __name__ == "__main__":
    main()
