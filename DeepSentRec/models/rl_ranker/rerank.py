#!/usr/bin/env python
"""
Module 4 — Apply trained PPO ranker to rerank a candidate list.

Input:
- ppo model
- candidates as json: [{"item": int, "base_score": float, "sentiment_score": float}, ...]
Output:
- reranked candidates

This is used to integrate RLRanker-PPO on top of HybridCF-SBERT in inference.
"""

from __future__ import annotations
import argparse, json
import numpy as np

from .environment import EnvConfig, RLRankerEnv

def build_single_episode(candidates, steps=1):
    ep = {"user_idx": 0, "steps": []}
    for _ in range(steps):
        ep["steps"].append({"candidates": candidates, "clicked_item": None, "purchased_item": None})
    return [ep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ppo_ranker.zip")
    ap.add_argument("--candidates_json", required=True)
    ap.add_argument("--k_candidates", type=int, default=20)
    ap.add_argument("--action_type", choices=["swap","rotate"], default="swap")
    ap.add_argument("--steps", type=int, default=10, help="Number of actions to apply.")
    args = ap.parse_args()

    candidates = json.loads(args.candidates_json)
    cfg = EnvConfig(k_candidates=args.k_candidates, max_steps_per_episode=args.steps, action_type=args.action_type)
    env = RLRankerEnv(build_single_episode(candidates, steps=args.steps), cfg)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(args.model)

    obs = vec_env.reset()
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if done:
            break

    # Extract final candidate list from underlying env
    final_cand = vec_env.envs[0]._current["candidates"]
    print(json.dumps(final_cand))

if __name__ == "__main__":
    main()
