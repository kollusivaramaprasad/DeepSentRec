"""
SentimentBERT configuration for DeepSentRec.

Implements the methodology:
- Base encoder: DistilBERT (small footprint).
- Stage 1: Pre-train / warm-start on IMDB (binary).
- Stage 2: Fine-tune on Amazon + Yelp (3-class: negative/neutral/positive).
"""

from dataclasses import dataclass

@dataclass
class SentimentConfig:
    base_model_name: str = "distilbert-base-uncased"
    max_length: int = 192
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    seed: int = 42

    # PPO / ranking uses different gamma etc., not here.

# 3-class label space for DeepSentRec
LABEL2ID_3 = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL_3 = {v: k for k, v in LABEL2ID_3.items()}

# 2-class for IMDB stage (mapped later to 3-class for finetuning)
LABEL2ID_2 = {"negative": 0, "positive": 1}
ID2LABEL_2 = {v: k for k, v in LABEL2ID_2.items()}
