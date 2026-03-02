#!/usr/bin/env python
"""
Utility — user-stratified train/valid/test split.

Given interactions with columns user_id, item_id, rating:
- For each user, hold out last N interactions for test (by timestamp if available),
  else random.
- Optionally also hold out for validation.

Outputs:
- interactions_train.parquet
- interactions_valid.parquet (optional)
- interactions_test.parquet
"""

from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="interactions.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_n", type=int, default=1)
    ap.add_argument("--valid_n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.lower().endswith(".parquet") else pd.read_csv(args.input)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df = df.sort_values(["user_id","timestamp"])
    else:
        df = df.sample(frac=1.0, random_state=args.seed).sort_values(["user_id"])

    train_rows = []
    valid_rows = []
    test_rows = []

    for uid, g in df.groupby("user_id"):
        g = g.reset_index(drop=True)
        n = len(g)
        t_n = min(args.test_n, max(n-1, 0))
        v_n = min(args.valid_n, max(n-1-t_n, 0))
        if t_n > 0:
            test = g.tail(t_n)
            rest = g.iloc[: n - t_n]
        else:
            test = g.iloc[0:0]
            rest = g
        if v_n > 0:
            valid = rest.tail(v_n)
            train = rest.iloc[: len(rest) - v_n]
        else:
            valid = rest.iloc[0:0]
            train = rest

        train_rows.append(train)
        valid_rows.append(valid)
        test_rows.append(test)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else df.iloc[0:0]
    valid_df = pd.concat(valid_rows, ignore_index=True) if valid_rows else df.iloc[0:0]
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else df.iloc[0:0]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "interactions_train.parquet")
    test_path = os.path.join(args.out_dir, "interactions_test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    if args.valid_n > 0:
        valid_path = os.path.join(args.out_dir, "interactions_valid.parquet")
        valid_df.to_parquet(valid_path, index=False)

    print(f"Train={len(train_df)} Valid={len(valid_df)} Test={len(test_df)} -> {args.out_dir}")

if __name__ == "__main__":
    main()
