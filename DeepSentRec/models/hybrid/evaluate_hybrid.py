#!/usr/bin/env python
"""
Module 3 — Evaluate HybridCF-SBERT on a user-stratified split.

Input:
- interactions.parquet (train) and interactions_test.parquet (test)
- hybrid artifacts folder (from train_hybrid.py)

We treat test as held-out positive interactions per user.
For each user, recommend top-K items and compute metrics.

Note:
- For large datasets, sample users via --limit_users.
"""

from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd

from .hybrid_scorer import recommend_topk
from DeepSentRec.evaluation.recommendation_metrics import precision_at_k, hit_rate_at_k, ndcg_at_k, average_precision_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_interactions", required=True)
    ap.add_argument("--test_interactions", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit_users", type=int, default=0)
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train_interactions)
    test_df = pd.read_parquet(args.test_interactions)

    users = np.load(os.path.join(args.artifacts, "users.npy"), allow_pickle=True)
    items = np.load(os.path.join(args.artifacts, "items.npy"), allow_pickle=True)
    W = np.load(os.path.join(args.artifacts, "nmf_W.npy"))
    H = np.load(os.path.join(args.artifacts, "nmf_H.npy"))
    item_ids = np.load(os.path.join(args.artifacts, "item_ids.npy"), allow_pickle=True)
    item_emb = np.load(os.path.join(args.artifacts, "item_embeddings.npy"))
    user_profiles = np.load(os.path.join(args.artifacts, "user_profiles.npy"))

    # Masks of seen items per user (train)
    item_index = {it:i for i,it in enumerate(items)}
    user_index = {u:i for i,u in enumerate(users)}
    seen = [set() for _ in range(len(users))]
    for r in train_df.itertuples(index=False):
        u = getattr(r, "user_id")
        it = getattr(r, "item_id")
        if u in user_index and it in item_index:
            seen[user_index[u]].add(item_index[it])

    # Relevant items per user from test
    rel = [set() for _ in range(len(users))]
    for r in test_df.itertuples(index=False):
        u = getattr(r, "user_id")
        it = getattr(r, "item_id")
        if u in user_index and it in item_index:
            rel[user_index[u]].add(item_index[it])

    user_ids = list(range(len(users)))
    if args.limit_users and args.limit_users > 0:
        user_ids = user_ids[:args.limit_users]

    ps, hrs, ndcgs, aps = [], [], [], []
    for ui in user_ids:
        if len(rel[ui]) == 0:
            continue
        mask = np.zeros((len(items),), dtype=bool)
        if seen[ui]:
            mask[list(seen[ui])] = True

        top_idx, _ = recommend_topk(
            user_idx=ui,
            W=W,
            H=H,
            user_profile=user_profiles[ui],
            item_emb=item_emb if len(item_emb)==len(items) else item_emb,  # assumes aligned to items
            alpha=args.alpha,
            k=args.k,
            seen_item_mask=mask
        )
        rec_list = top_idx.tolist()
        rel_set = rel[ui]

        ps.append(precision_at_k(rec_list, rel_set, args.k))
        hrs.append(hit_rate_at_k(rec_list, rel_set, args.k))
        ndcgs.append(ndcg_at_k(rec_list, rel_set, args.k))
        aps.append(average_precision_at_k(rec_list, rel_set, args.k))

    print(f"Users evaluated: {len(ps)}")
    print(f"Precision@{args.k}: {float(np.mean(ps)):.4f}")
    print(f"HitRate@{args.k}: {float(np.mean(hrs)):.4f}")
    print(f"NDCG@{args.k}: {float(np.mean(ndcgs)):.4f}")
    print(f"MAP@{args.k}: {float(np.mean(aps)):.4f}")

if __name__ == "__main__":
    main()
