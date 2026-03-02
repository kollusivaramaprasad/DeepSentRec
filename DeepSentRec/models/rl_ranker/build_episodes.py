#!/usr/bin/env python
"""
Module 4 — Build offline RL episodes from logs + HybridCF-SBERT candidates.

Input:
- interactions parquet with columns: user_id, item_id, rating (and optional click/purchase)
- hybrid artifacts (Module 3): users.npy, items.npy, nmf_W.npy, nmf_H.npy, user_profiles.npy, item_embeddings.npy
- candidate generation uses Hybrid scorer (alpha) to get top-K candidates per user per timestep.

Offline simplification:
- For each user, we create an "episode" consisting of T timesteps.
- Each timestep uses the same candidate list (can be extended to time-aware logs).
- clicked_item / purchased_item come from held-out interactions.

Outputs:
- episodes.jsonl containing episodes that RLRankerEnv can load.

This is a pragmatic offline setup aligned with your methodology's offline simulation.
"""

from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

from DeepSentRec.models.hybrid.hybrid_scorer import recommend_topk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_interactions", required=True, help="Train interactions parquet.")
    ap.add_argument("--test_interactions", required=True, help="Test interactions parquet (positives).")
    ap.add_argument("--hybrid_artifacts", required=True, help="Module3 artifacts directory.")
    ap.add_argument("--out", required=True, help="episodes.jsonl")
    ap.add_argument("--k_candidates", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max_users", type=int, default=2000)
    ap.add_argument("--steps_per_user", type=int, default=10)
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train_interactions)
    test_df = pd.read_parquet(args.test_interactions)

    users = np.load(os.path.join(args.hybrid_artifacts,"users.npy"), allow_pickle=True)
    items = np.load(os.path.join(args.hybrid_artifacts,"items.npy"), allow_pickle=True)
    W = np.load(os.path.join(args.hybrid_artifacts,"nmf_W.npy"))
    H = np.load(os.path.join(args.hybrid_artifacts,"nmf_H.npy"))
    user_profiles = np.load(os.path.join(args.hybrid_artifacts,"user_profiles.npy"))
    item_emb = np.load(os.path.join(args.hybrid_artifacts,"item_embeddings.npy"))

    user_index = {u:i for i,u in enumerate(users)}
    item_index = {it:i for i,it in enumerate(items)}

    # Seen masks from train
    seen = [set() for _ in range(len(users))]
    for r in train_df.itertuples(index=False):
        u = getattr(r,"user_id")
        it = getattr(r,"item_id")
        if u in user_index and it in item_index:
            seen[user_index[u]].add(item_index[it])

    # Positives per user from test (we use first as clicked target)
    positives = [None for _ in range(len(users))]
    for r in test_df.itertuples(index=False):
        u = getattr(r,"user_id")
        it = getattr(r,"item_id")
        if u in user_index and it in item_index and positives[user_index[u]] is None:
            positives[user_index[u]] = item_index[it]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ui in range(len(users)):
            if written >= args.max_users:
                break
            pos = positives[ui]
            if pos is None:
                continue

            mask = np.zeros((len(items),), dtype=bool)
            if seen[ui]:
                mask[list(seen[ui])] = True

            top_idx, top_scores = recommend_topk(
                user_idx=ui,
                W=W,
                H=H,
                user_profile=user_profiles[ui],
                item_emb=item_emb,
                alpha=args.alpha,
                k=args.k_candidates,
                seen_item_mask=mask
            )

            # Build candidate dicts with base_score; sentiment_score optional (0 for now)
            candidates = [{"item": int(i), "base_score": float(s), "sentiment_score": 0.0} for i, s in zip(top_idx, top_scores)]
            steps = []
            for _t in range(args.steps_per_user):
                steps.append({
                    "candidates": candidates,
                    "clicked_item": int(pos),
                    "purchased_item": None
                })
            episode = {"user_idx": int(ui), "steps": steps}
            f.write(json.dumps(episode) + "\n")
            written += 1

    print(f"Wrote episodes: {written} -> {out_path}")

if __name__ == "__main__":
    main()
