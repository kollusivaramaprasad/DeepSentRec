"""
Module 3 — Collaborative Filtering via NMF.

Uses sklearn.decomposition.NMF to factorize the user-item matrix.
Outputs latent matrices W (user) and H (item) and predicted scores.

This aligns with the methodology NMF factorization U ≈ W H.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF

def make_sparse_matrix(interactions: pd.DataFrame, users: np.ndarray, items: np.ndarray):
    user_index = {u:i for i,u in enumerate(users)}
    item_index = {it:i for i,it in enumerate(items)}
    rows = interactions["user_id"].map(user_index).to_numpy()
    cols = interactions["item_id"].map(item_index).to_numpy()
    vals = interactions["rating"].astype(float).to_numpy()
    mat = coo_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return mat.tocsr()

def train_nmf(interactions: pd.DataFrame, users: np.ndarray, items: np.ndarray, k: int = 50, seed: int = 42, max_iter: int = 200):
    X = make_sparse_matrix(interactions, users, items)
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=max_iter)
    W = model.fit_transform(X)
    H = model.components_  # [k, n_items]
    return model, W, H

def predict_scores_for_user(W: np.ndarray, H: np.ndarray, user_idx: int):
    # score = W[u] @ H
    return W[user_idx].dot(H)
