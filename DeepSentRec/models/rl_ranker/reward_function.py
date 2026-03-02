"""
Module 4 — RLRanker-PPO reward function.

Matches the methodology:
R_t = λ1 * CTR_t + λ2 * Purchase_t - λ3 * Skip_t

In offline training, we approximate these from logged interactions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class RewardWeights:
    w_click: float = 1.0
    w_purchase: float = 3.0
    w_skip: float = 1.0

def compute_reward(feedback: Dict[str, float], w: RewardWeights) -> float:
    """
    feedback keys (any subset):
      - click: 0/1
      - purchase: 0/1
      - skip: 0/1
      - dwell: optional (not used by default)
    """
    click = float(feedback.get("click", 0.0))
    purchase = float(feedback.get("purchase", 0.0))
    skip = float(feedback.get("skip", 0.0))
    return w.w_click * click + w.w_purchase * purchase - w.w_skip * skip
