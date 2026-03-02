#!/usr/bin/env python
"""
Minimal Flask API for DeepSentRec demo.

Endpoints:
- /health
- /recommend?user_index=0

Assumes artifacts exist:
- artifacts/hybrid/*
- artifacts/rl/ppo_ranker.zip
"""

from __future__ import annotations
import json, os
import numpy as np
from flask import Flask, request, jsonify

from DeepSentRec.models.hybrid.hybrid_scorer import recommend_topk
from DeepSentRec.models.rl_ranker.environment import EnvConfig, RLRankerEnv

app = Flask(__name__)

# Lazy-loaded globals
g = {}

def load_artifacts(hybrid_dir: str, ppo_path: str):
    if g.get("loaded"):
        return
    g["users"] = np.load(os.path.join(hybrid_dir,"users.npy"), allow_pickle=True)
    g["items"] = np.load(os.path.join(hybrid_dir,"items.npy"), allow_pickle=True)
    g["W"] = np.load(os.path.join(hybrid_dir,"nmf_W.npy"))
    g["H"] = np.load(os.path.join(hybrid_dir,"nmf_H.npy"))
    g["user_profiles"] = np.load(os.path.join(hybrid_dir,"user_profiles.npy"))
    g["item_emb"] = np.load(os.path.join(hybrid_dir,"item_embeddings.npy"))

    from stable_baselines3 import PPO
    g["ppo"] = PPO.load(ppo_path)
    g["loaded"] = True

@app.get("/health")
def health():
    return jsonify({"status":"ok"})

@app.get("/recommend")
def recommend():
    hybrid_dir = request.args.get("hybrid_dir","artifacts/hybrid")
    ppo_path = request.args.get("ppo_model","artifacts/rl/ppo_ranker.zip")
    user_index = int(request.args.get("user_index","0"))
    k_candidates = int(request.args.get("k_candidates","20"))
    alpha = float(request.args.get("alpha","0.5"))
    steps = int(request.args.get("steps","10"))

    load_artifacts(hybrid_dir, ppo_path)

    users, items, W, H, prof, emb = g["users"], g["items"], g["W"], g["H"], g["user_profiles"], g["item_emb"]

    top_idx, top_scores = recommend_topk(user_index, W, H, prof[user_index], emb, alpha=alpha, k=k_candidates)
    candidates = [{"item": int(i), "base_score": float(s), "sentiment_score": 0.0} for i, s in zip(top_idx, top_scores)]

    from stable_baselines3.common.vec_env import DummyVecEnv
    cfg = EnvConfig(k_candidates=k_candidates, max_steps_per_episode=steps, action_type="swap")
    env = RLRankerEnv([{"user_idx": 0, "steps":[{"candidates": candidates, "clicked_item": None, "purchased_item": None} for _ in range(steps)]}], cfg)
    vec = DummyVecEnv([lambda: env])
    obs = vec.reset()
    for _ in range(steps):
        action, _ = g["ppo"].predict(obs, deterministic=True)
        obs, reward, done, info = vec.step(action)
        if done:
            break

    reranked = vec.envs[0]._current["candidates"]
    out_items = [items[c["item"]] if 0 <= c["item"] < len(items) else None for c in reranked]
    return jsonify({"user": str(users[user_index]), "recommendations": out_items[:10]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
