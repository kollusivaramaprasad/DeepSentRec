"""
Dataset utilities for SentimentBERT.

Inputs:
- Standardized parquet/csv built in Module 1.
Expected columns (minimum):
  - review_text (str)
Optional:
  - rating (float/int): used to derive 3-class sentiment labels:
      1-2 -> negative, 3 -> neutral, 4-5 -> positive
  - sentiment_label (str/int): if already present, used directly
  - source: helps debugging

For IMDB HuggingFace dataset stage, use build_dataset.py (Module 1) to export parquet,
then point these utilities to that parquet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import LABEL2ID_3, LABEL2ID_2


def derive_3class_from_rating(rating: float) -> Optional[int]:
    """Map star rating to 3-class sentiment ID."""
    if rating is None:
        return None
    try:
        r = float(rating)
    except Exception:
        return None
    if r <= 2.0:
        return LABEL2ID_3["negative"]
    if 2.0 < r < 4.0:  # typically 3
        return LABEL2ID_3["neutral"]
    return LABEL2ID_3["positive"]


def derive_2class_from_imdb_label(label: int) -> int:
    """IMDB label: 0=neg, 1=pos"""
    return int(label)


@dataclass
class DataFrameSentimentSpec:
    text_col: str = "review_text"
    rating_col: str = "rating"
    sentiment_col: str = "sentiment_label"


class SentimentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        label_mode: str = "3class",
        spec: DataFrameSentimentSpec = DataFrameSentimentSpec(),
    ):
        assert label_mode in {"3class", "2class"}, "label_mode must be '3class' or '2class'"
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.label_mode = label_mode
        self.spec = spec

        # Ensure text exists
        if self.spec.text_col not in self.df.columns:
            raise ValueError(f"Missing required text column: {self.spec.text_col}")

        self._labels = self._build_labels()

    def _build_labels(self) -> List[int]:
        labels: List[int] = []
        if self.label_mode == "3class":
            for _, row in self.df.iterrows():
                # Prefer existing sentiment_label if present
                if self.spec.sentiment_col in self.df.columns and pd.notna(row.get(self.spec.sentiment_col)):
                    lab = row.get(self.spec.sentiment_col)
                    if isinstance(lab, str):
                        lab = lab.strip().lower()
                        if lab not in LABEL2ID_3:
                            raise ValueError(f"Unknown sentiment label string: {lab}")
                        labels.append(LABEL2ID_3[lab])
                    else:
                        labels.append(int(lab))
                    continue
                # Otherwise derive from rating
                lab = derive_3class_from_rating(row.get(self.spec.rating_col))
                if lab is None:
                    # If no rating, default neutral (safer) — or filter upstream
                    lab = LABEL2ID_3["neutral"]
                labels.append(int(lab))
        else:
            # 2-class mode: expects sentiment_label as 0/1 or rating mapped to neg/pos
            for _, row in self.df.iterrows():
                lab = None
                if self.spec.sentiment_col in self.df.columns and pd.notna(row.get(self.spec.sentiment_col)):
                    lab = int(row.get(self.spec.sentiment_col))
                else:
                    # derive from rating if present (<=3 -> neg, >=4 -> pos)
                    r = row.get(self.spec.rating_col)
                    if r is not None and pd.notna(r):
                        lab = 0 if float(r) <= 3.0 else 1
                    else:
                        lab = 0
                labels.append(int(lab))
        return labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.df.loc[idx, self.spec.text_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


def load_parquet_or_csv(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)
