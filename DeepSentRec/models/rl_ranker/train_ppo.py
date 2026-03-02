#!/usr/bin/env python
"""
Module 4 — Train RLRanker-PPO using Stable-Baselines3.

This trains a PPO agent to select reranking actions in RLRankerEnv.

Inputs:
- episodes.jsonl (built by build_episodes.py)

Outputs:
- ppo_model.zip
- training logs

Note:
This is an offline simulation; for real deployment you'd do online fine-tuning with live feedback.
"""

from __future__ import annotations
import argparse, json, os
from typing import List, Dict, Any

import numpy as np

from .environment import RLRankerEnv, EnvConfig
from .reward_function import RewardWeights

def load_episodes(path: str) -> List[Dict[str, Any]]:
    eps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eps.append(json.loads(line))
    return eps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_candidates", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--action_type", choices=["swap","rotate"], default="swap")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    # reward weights
    ap.add_argument("--w_click", type=float, default=1.0)
    ap.add_argument("--w_purchase", type=float, default=3.0)
    ap.add_argument("--w_skip", type=float, default=1.0)
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    cfg = EnvConfig(
        k_candidates=args.k_candidates,
        max_steps_per_episode=args.max_steps,
        action_type=args.action_type,
        seed=args.seed,
        reward_weights=RewardWeights(args.w_click, args.w_purchase, args.w_skip),
    )
    env = RLRankerEnv(episodes, cfg)

    # SB3
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        clip_range=args.clip_range,
        batch_size=args.batch_size,
        verbose=1,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model.learn(total_timesteps=args.timesteps)

    out_model = os.path.join(args.out_dir, "ppo_ranker.zip")
    model.save(out_model)
    print(f"Saved PPO model: {out_model}")

if __name__ == "__main__":
    main()
