"""
Module 3 — Recommendation metrics (Precision@K, HitRate@K, NDCG@K, MAP@K).

These align with the methodology evaluation section.
"""

from __future__ import annotations
import numpy as np

def precision_at_k(recommended: list, relevant_set: set, k: int) -> float:
    rec_k = recommended[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for x in rec_k if x in relevant_set)
    return hits / k

def hit_rate_at_k(recommended: list, relevant_set: set, k: int) -> float:
    rec_k = recommended[:k]
    return 1.0 if any(x in relevant_set for x in rec_k) else 0.0

def dcg_at_k(recommended: list, relevant_set: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    return dcg

def ndcg_at_k(recommended: list, relevant_set: set, k: int) -> float:
    dcg = dcg_at_k(recommended, relevant_set, k)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision_at_k(recommended: list, relevant_set: set, k: int) -> float:
    hits = 0
    ap = 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            hits += 1
            ap += hits / i
    denom = min(len(relevant_set), k)
    return ap / denom if denom > 0 else 0.0

def map_at_k(recs: list[list], relevants: list[set], k: int) -> float:
    aps = [average_precision_at_k(r, rel, k) for r, rel in zip(recs, relevants)]
    return float(np.mean(aps)) if aps else 0.0
