"""
Module 3 — HybridCF-SBERT scorer.

Hybrid score:
S_hyb(u, j) = alpha * S_cf(u, j) + (1-alpha) * S_cbf(u, j)

Where:
- S_cf from NMF latent factors W,H.
- S_cbf from cosine similarity between item embeddings and user's profile embedding.
User profile embedding: mean of embeddings of items the user interacted with (positive history).

Also supports sentiment reweighting if sentiment_score is available in interactions table
(used as a prior; higher positive sentiment boosts the item).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def build_user_profiles(interactions: pd.DataFrame, users: np.ndarray, item_ids: np.ndarray, item_emb: np.ndarray):
    item_index = {it:i for i,it in enumerate(item_ids)}
    user_profiles = np.zeros((len(users), item_emb.shape[1]), dtype=np.float32)
    counts = np.zeros((len(users),), dtype=np.int32)
    user_index = {u:i for i,u in enumerate(users)}
    for row in interactions.itertuples(index=False):
        u = getattr(row, "user_id")
        it = getattr(row, "item_id")
        if it not in item_index:
            continue
        ui = user_index[u]
        user_profiles[ui] += item_emb[item_index[it]]
        counts[ui] += 1
    counts = np.maximum(counts, 1)
    user_profiles = user_profiles / counts[:, None]
    # Normalize
    norms = np.linalg.norm(user_profiles, axis=1, keepdims=True) + 1e-12
    user_profiles = user_profiles / norms
    return user_profiles

def recommend_topk(
    user_idx: int,
    W: np.ndarray,
    H: np.ndarray,
    user_profile: np.ndarray,
    item_emb: np.ndarray,
    alpha: float = 0.5,
    k: int = 10,
    seen_item_mask: np.ndarray | None = None,
    sentiment_item_prior: np.ndarray | None = None,
):
    cf_scores = W[user_idx].dot(H)  # [n_items]
    # content scores: cosine between user profile and each item embedding (both normalized)
    cbf_scores = item_emb.dot(user_profile)  # [n_items]
    scores = alpha * cf_scores + (1.0 - alpha) * cbf_scores

    if sentiment_item_prior is not None:
        # prior expected in [0,1]; add small boost
        scores = scores + 0.1 * sentiment_item_prior

    if seen_item_mask is not None:
        scores = scores.copy()
        scores[seen_item_mask] = -1e9

    top_idx = np.argpartition(-scores, kth=min(k, len(scores)-1))[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx, scores[top_idx]
