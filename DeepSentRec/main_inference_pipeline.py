#!/usr/bin/env python
"""
End-to-end inference pipeline (HybridCF-SBERT + RLRanker-PPO).

Given:
- hybrid artifacts (Module 3)
- PPO model (Module 4)
- a user_id and their history (interactions table)

Produces:
- top-N reranked item indices (or item_ids if mapping present)

This is a research-grade inference script (not a web service).
"""

from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

from DeepSentRec.models.hybrid.hybrid_scorer import recommend_topk
from DeepSentRec.models.rl_ranker.environment import EnvConfig, RLRankerEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid_artifacts", required=True)
    ap.add_argument("--ppo_model", required=True)
    ap.add_argument("--user_index", type=int, required=True, help="User index in users.npy")
    ap.add_argument("--k_candidates", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    users = np.load(os.path.join(args.hybrid_artifacts,"users.npy"), allow_pickle=True)
    items = np.load(os.path.join(args.hybrid_artifacts,"items.npy"), allow_pickle=True)
    W = np.load(os.path.join(args.hybrid_artifacts,"nmf_W.npy"))
    H = np.load(os.path.join(args.hybrid_artifacts,"nmf_H.npy"))
    user_profiles = np.load(os.path.join(args.hybrid_artifacts,"user_profiles.npy"))
    item_emb = np.load(os.path.join(args.hybrid_artifacts,"item_embeddings.npy"))

    ui = int(args.user_index)
    top_idx, top_scores = recommend_topk(
        user_idx=ui, W=W, H=H,
        user_profile=user_profiles[ui],
        item_emb=item_emb,
        alpha=args.alpha,
        k=args.k_candidates,
        seen_item_mask=None
    )
    candidates = [{"item": int(i), "base_score": float(s), "sentiment_score": 0.0} for i, s in zip(top_idx, top_scores)]

    # RL rerank
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = EnvConfig(k_candidates=args.k_candidates, max_steps_per_episode=args.steps, action_type="swap")
    env = RLRankerEnv([{"user_idx": 0, "steps":[{"candidates": candidates, "clicked_item": None, "purchased_item": None} for _ in range(args.steps)]}], cfg)
    vec = DummyVecEnv([lambda: env])
    model = PPO.load(args.ppo_model)

    obs = vec.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec.step(action)
        if done:
            break

    reranked = vec.envs[0]._current["candidates"]
    # map to item_id strings if possible
    out_items = [items[c["item"]] if 0 <= c["item"] < len(items) else None for c in reranked]
    print(json.dumps({"user": str(users[ui]), "reranked_item_ids": out_items[:10]}))

if __name__ == "__main__":
    main()
