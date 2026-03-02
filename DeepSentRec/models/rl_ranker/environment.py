#!/usr/bin/env python
"""
Module 4 — RLRanker-PPO Environment (offline) for reranking.

We model reranking as an MDP:
- State s_t: a vector describing the current candidate list for a user at timestep t.
- Action a_t: a reranking action applied to the candidate list (swap / shift / top-k permutation).
- Reward r_t: computed from user feedback (click/purchase/skip) on the item shown at top.

Offline approximation:
- We replay logged interactions: for each (user, timestep), we have:
    - candidate_items: list of item indices (from HybridCF-SBERT)
    - clicked_item (optional) and purchased_item (optional)
- The environment applies the action to reorder the candidate list and then
  assigns reward based on whether the top item matches the logged positive item.

This is a standard offline-to-online bridge and is consistent with the paper's
"simulated mock-user interactions offline".

Dependencies:
- gymnasium (preferred) or gym
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # fallback
    import gym
    from gym import spaces

from .reward_function import RewardWeights, compute_reward


@dataclass
class EnvConfig:
    k_candidates: int = 20
    max_steps_per_episode: int = 50
    action_type: str = "swap"  # "swap" or "rotate"
    seed: int = 42
    reward_weights: RewardWeights = RewardWeights()


class RLRankerEnv(gym.Env):
    """
    Observation: concatenated features for k candidates:
      - base_score (from Hybrid model) [k]
      - sentiment_score (optional) [k]
    Shape: (k, f) flattened to (k*f,)
    Action space:
      - swap: Discrete(k*k) representing swap(i,j)
      - rotate: Discrete(k) representing rotate top item to position p
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        config: EnvConfig,
    ):
        super().__init__()
        self.episodes = episodes
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

        self.k = int(self.cfg.k_candidates)
        self.f = 2  # base_score + sentiment_score (default; can be extended)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.k * self.f,), dtype=np.float32
        )

        if self.cfg.action_type == "swap":
            self.action_space = spaces.Discrete(self.k * self.k)
        elif self.cfg.action_type == "rotate":
            self.action_space = spaces.Discrete(self.k)
        else:
            raise ValueError("action_type must be 'swap' or 'rotate'")

        self._episode_idx = -1
        self._step = 0
        self._current = None

    def _get_obs(self) -> np.ndarray:
        cand = self._current["candidates"]  # list of dicts
        # Ensure length k
        cand = cand[: self.k]
        while len(cand) < self.k:
            cand.append({"base_score": 0.0, "sentiment_score": 0.0, "item": -1})
        base = np.array([c.get("base_score", 0.0) for c in cand], dtype=np.float32)
        sent = np.array([c.get("sentiment_score", 0.0) for c in cand], dtype=np.float32)
        feat = np.stack([base, sent], axis=1).reshape(-1)
        return feat

    def _apply_action(self, action: int) -> None:
        cand = self._current["candidates"]
        cand = cand[: self.k]
        if self.cfg.action_type == "swap":
            i = action // self.k
            j = action % self.k
            if i < len(cand) and j < len(cand):
                cand[i], cand[j] = cand[j], cand[i]
        else:  # rotate
            p = int(action)
            if 0 <= p < len(cand):
                top = cand[0]
                rest = cand[1:]
                # insert top at position p
                rest.insert(p, top)
                cand[:] = rest[: len(cand)]
        self._current["candidates"] = cand

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._episode_idx = (self._episode_idx + 1) % len(self.episodes)
        self._current = {
            "steps": self.episodes[self._episode_idx]["steps"],
        }
        self._step = 0
        return self._step_state(), {}

    def _step_state(self):
        # Each episode is a sequence of timesteps; pick current
        steps = self._current["steps"]
        t = min(self._step, len(steps) - 1)
        self._current["timestep"] = steps[t]
        self._current["candidates"] = list(self._current["timestep"]["candidates"])
        return self._get_obs()

    def step(self, action: int):
        # Apply reranking action
        self._apply_action(int(action))

        # Feedback is computed w.r.t. logged positive item for this timestep
        ts = self._current["timestep"]
        clicked_item = ts.get("clicked_item", None)
        purchased_item = ts.get("purchased_item", None)

        top_item = self._current["candidates"][0].get("item", None)

        feedback = {"click": 0.0, "purchase": 0.0, "skip": 0.0}
        if purchased_item is not None and top_item == purchased_item:
            feedback["purchase"] = 1.0
            feedback["click"] = 1.0
        elif clicked_item is not None and top_item == clicked_item:
            feedback["click"] = 1.0
        else:
            feedback["skip"] = 1.0

        reward = compute_reward(feedback, self.cfg.reward_weights)

        self._step += 1
        terminated = self._step >= min(self.cfg.max_steps_per_episode, len(self._current["steps"]))
        truncated = False

        obs = self._step_state() if not terminated else self._get_obs()
        info = {"feedback": feedback, "top_item": top_item}
        return obs, float(reward), terminated, truncated, info
