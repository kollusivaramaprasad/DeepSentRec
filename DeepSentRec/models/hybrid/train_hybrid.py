#!/usr/bin/env python
"""
Module 3 — Train HybridCF-SBERT.

Inputs:
- interactions.parquet + mappings.npz from build_interactions.py
- item embeddings from sbert_items.py (item_ids.npy + item_embeddings.npy)

Outputs (out_dir):
- nmf_W.npy, nmf_H.npy
- users.npy, items.npy
- item_ids.npy, item_embeddings.npy (copied)
- user_profiles.npy
"""

from __future__ import annotations
import argparse, os, shutil
import numpy as np
import pandas as pd

from .nmf_model import train_nmf
from .hybrid_scorer import build_user_profiles

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--mappings", required=True)
    ap.add_argument("--item_ids", required=True)
    ap.add_argument("--item_embeddings", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inter = pd.read_parquet(args.interactions)
    maps = np.load(args.mappings, allow_pickle=True)
    users = maps["users"]
    items = maps["items"]

    item_ids = np.load(args.item_ids, allow_pickle=True)
    item_emb = np.load(args.item_embeddings)

    model, W, H = train_nmf(inter, users, items, k=args.k, seed=args.seed, max_iter=args.max_iter)

    # Build user profiles (content side)
    user_profiles = build_user_profiles(inter, users, item_ids, item_emb)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "users.npy"), users)
    np.save(os.path.join(args.out_dir, "items.npy"), items)
    np.save(os.path.join(args.out_dir, "nmf_W.npy"), W.astype(np.float32))
    np.save(os.path.join(args.out_dir, "nmf_H.npy"), H.astype(np.float32))
    np.save(os.path.join(args.out_dir, "user_profiles.npy"), user_profiles.astype(np.float32))

    # copy item embedding artifacts
    shutil.copy2(args.item_ids, os.path.join(args.out_dir, "item_ids.npy"))
    shutil.copy2(args.item_embeddings, os.path.join(args.out_dir, "item_embeddings.npy"))

    print(f"Saved hybrid artifacts to: {args.out_dir}")
    print(f"W={W.shape} H={H.shape} user_profiles={user_profiles.shape} item_emb={item_emb.shape}")

if __name__ == "__main__":
    main()
