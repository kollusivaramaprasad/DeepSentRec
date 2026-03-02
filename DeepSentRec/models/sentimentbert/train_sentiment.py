#!/usr/bin/env python
"""
Module 2: SentimentBERT training for DeepSentRec.

Implements:
- DistilBERT-based sentiment classifier.
- Optional 2-stage training:
  (A) stage1_imdb: 2-class warm-start
  (B) stage2_finetune: 3-class fine-tuning on Amazon+Yelp (rating->3class)

Outputs:
- saved model + tokenizer in output_dir
- training logs
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .config import SentimentConfig, LABEL2ID_3, ID2LABEL_3, LABEL2ID_2, ID2LABEL_2
from .dataset import SentimentDataset, load_parquet_or_csv


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / max(total, 1)


def train_one_stage(
    stage_name: str,
    train_path: str,
    valid_path: str,
    output_dir: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    base_model_name: str,
    max_length: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    seed: int,
    device: str,
    resume_from: str | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(resume_from or base_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        resume_from or base_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model.to(device)

    train_df = load_parquet_or_csv(train_path)
    valid_df = load_parquet_or_csv(valid_path) if valid_path else None

    label_mode = "2class" if num_labels == 2 else "3class"
    train_ds = SentimentDataset(train_df, tokenizer, max_length=max_length, label_mode=label_mode)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    valid_dl = None
    if valid_df is not None:
        valid_ds = SentimentDataset(valid_df, tokenizer, max_length=max_length, label_mode=label_mode)
        valid_dl = DataLoader(valid_ds, batch_size=eval_batch_size, shuffle=False, num_workers=2)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=lr)

    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()

        val_acc = None
        if valid_dl is not None:
            val_acc = evaluate(model, valid_dl, device)

        # Save checkpoint
        ckpt_dir = os.path.join(output_dir, f"{stage_name}_epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        if val_acc is not None and val_acc > best_acc:
            best_acc = val_acc
            best_dir = os.path.join(output_dir, f"{stage_name}_best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        print(f"[{stage_name}] epoch={epoch}/{epochs} loss={running_loss/len(train_dl):.4f} val_acc={val_acc}")

    # Return best if exists else last
    best_path = os.path.join(output_dir, f"{stage_name}_best")
    return best_path if os.path.exists(best_path) else ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1_imdb", "stage2_finetune", "single_3class"], required=True)
    ap.add_argument("--train", required=True, help="Path to train parquet/csv.")
    ap.add_argument("--valid", default="", help="Path to valid parquet/csv (optional).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resume_from", default="", help="Optional model dir for warm-start.")
    ap.add_argument("--base_model", default=SentimentConfig().base_model_name)
    ap.add_argument("--max_length", type=int, default=SentimentConfig().max_length)
    ap.add_argument("--lr", type=float, default=SentimentConfig().learning_rate)
    ap.add_argument("--weight_decay", type=float, default=SentimentConfig().weight_decay)
    ap.add_argument("--batch_size", type=int, default=SentimentConfig().train_batch_size)
    ap.add_argument("--eval_batch_size", type=int, default=SentimentConfig().eval_batch_size)
    ap.add_argument("--epochs", type=int, default=SentimentConfig().num_epochs)
    ap.add_argument("--seed", type=int, default=SentimentConfig().seed)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resume = args.resume_from.strip() or None

    if args.stage == "stage1_imdb":
        best = train_one_stage(
            stage_name="imdb_2class",
            train_path=args.train,
            valid_path=args.valid.strip() or "",
            output_dir=args.output_dir,
            num_labels=2,
            id2label=ID2LABEL_2,
            label2id=LABEL2ID_2,
            base_model_name=args.base_model,
            max_length=args.max_length,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
            resume_from=resume,
        )
        print(f"BEST_MODEL_DIR={best}")

    elif args.stage in {"stage2_finetune", "single_3class"}:
        # In stage2, resume_from should be stage1 best directory (2-class) OR base model.
        # Note: Transformers will adapt the classifier head if num_labels differs.
        best = train_one_stage(
            stage_name="finetune_3class",
            train_path=args.train,
            valid_path=args.valid.strip() or "",
            output_dir=args.output_dir,
            num_labels=3,
            id2label=ID2LABEL_3,
            label2id=LABEL2ID_3,
            base_model_name=args.base_model,
            max_length=args.max_length,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
            resume_from=resume,
        )
        print(f"BEST_MODEL_DIR={best}")

    else:
        raise ValueError("Unsupported stage")


if __name__ == "__main__":
    main()
