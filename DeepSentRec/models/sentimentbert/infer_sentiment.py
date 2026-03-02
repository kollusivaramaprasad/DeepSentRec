#!/usr/bin/env python
"""
Module 2: SentimentBERT inference for DeepSentRec.

Given a trained SentimentBERT directory (HF format) and a standardized dataset parquet/csv,
this script outputs:
- sentiment_label (string)
- sentiment_id (int)
- sentiment_score (max prob)
- sentiment_probs (optional json string)
- sentiment_embedding (vector) saved separately as .npy (for scale), with an index mapping.

Outputs:
- out_table: parquet/csv with added sentiment columns + embedding_index
- out_embeddings: .npy matrix [N, hidden_size]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import ID2LABEL_3, ID2LABEL_2
from .dataset import SentimentDataset, load_parquet_or_csv


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Trained SentimentBERT directory.")
    ap.add_argument("--data", required=True, help="Input parquet/csv built from Module 1.")
    ap.add_argument("--out_table", required=True, help="Output parquet/csv with sentiment fields.")
    ap.add_argument("--out_embeddings", required=True, help="Output .npy embeddings file.")
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--label_mode", choices=["3class", "2class"], default="3class")
    ap.add_argument("--store_probs", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_parquet_or_csv(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, output_hidden_states=True)
    model.to(device)
    model.eval()

    ds = SentimentDataset(df, tokenizer, max_length=args.max_length, label_mode=args.label_mode)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_ids = []
    all_scores = []
    all_labels = []
    all_probs = []
    embeds = []

    id2label = ID2LABEL_3 if args.label_mode == "3class" else ID2LABEL_2

    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}  # ignore labels on inference
        out = model(**batch)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1)
        pred_score = probs.max(dim=-1).values

        # CLS embedding from last hidden state: shape [B, T, H]
        # DistilBERT uses first token as CLS.
        last_hidden = out.hidden_states[-1]
        cls = last_hidden[:, 0, :].detach().cpu().numpy()

        embeds.append(cls)
        all_ids.extend(pred_id.detach().cpu().numpy().tolist())
        all_scores.extend(pred_score.detach().cpu().numpy().tolist())
        all_labels.extend([id2label[int(i)] for i in pred_id.detach().cpu().numpy().tolist()])

        if args.store_probs:
            all_probs.extend([json.dumps(p.tolist()) for p in probs.detach().cpu().numpy()])

    emb_mat = np.concatenate(embeds, axis=0)
    np.save(args.out_embeddings, emb_mat)

    out_df = df.copy()
    out_df["sentiment_id"] = all_ids
    out_df["sentiment_label"] = all_labels
    out_df["sentiment_score"] = all_scores
    out_df["embedding_index"] = np.arange(len(out_df), dtype=int)
    if args.store_probs:
        out_df["sentiment_probs"] = all_probs

    os.makedirs(os.path.dirname(args.out_table) or ".", exist_ok=True)
    if args.out_table.lower().endswith(".parquet"):
        out_df.to_parquet(args.out_table, index=False)
    else:
        out_df.to_csv(args.out_table, index=False)

    print(f"Saved table: {args.out_table}")
    print(f"Saved embeddings: {args.out_embeddings} shape={emb_mat.shape}")


if __name__ == "__main__":
    main()
