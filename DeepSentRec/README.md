# DeepSentRec — Complete Codebase (Modules 1–4)

This repository implements the methodology for:
**DeepSentRec: A Deep Learning-Based Sentiment-Aware Product Recommendation System**

Modules:
1. Dataset standardization + preprocessing
2. SentimentBERT (DistilBERT) sentiment extraction
3. HybridCF-SBERT (NMF collaborative filtering + SBERT content similarity)
4. RLRanker-PPO adaptive reranking (offline simulation)

> You must download datasets yourself and place them under `data/raw/` (or provide paths).


# DeepSentRec (Codebase Scaffold)

This repository implements **DeepSentRec: A Deep Learning-Based Sentiment-Aware Product Recommendation System**.

## What is implemented in this starter package?
**Module 1 (DONE): Data standardization**
- Loaders for: Amazon (json.gz), Yelp (json lines), IMDB (HuggingFace or csv), Kaggle e-commerce (csv, flexible column mapping)
- A unified schema (`utils/schema.py`) used by all downstream modules:
  - required: user_id, item_id, review_text, rating, timestamp, source
  - optional: review_id, sentiment_label, sentiment_score

This matches the methodology pipeline (SentimentBERT → HybridCF-SBERT → RLRanker-PPO).

## Quickstart

### 1) Create a virtual env and install deps
```bash
pip install -r requirements.txt
```

### 2) Build unified datasets
Amazon (UCSD/McAuley json.gz):
```bash
python build_dataset.py --dataset amazon --input data/raw/amazon_reviews.json.gz --out data/processed/amazon.parquet --limit 200000
```

Yelp review.json:
```bash
python build_dataset.py --dataset yelp --input data/raw/yelp_review.json --out data/processed/yelp.parquet --limit 200000
```

IMDB from HuggingFace:
```bash
python build_dataset.py --dataset imdb_hf --split train --out data/processed/imdb_train.parquet --limit 50000
```

Kaggle CSV:
```bash
python build_dataset.py --dataset kaggle --input data/raw/kaggle_ecom.csv --out data/processed/kaggle.parquet
```

## Next modules (to be implemented)
- Module 2: SentimentBERT training/inference scripts
- Module 3: HybridCF-SBERT (NMF + SBERT similarity + fusion)
- Module 4: RLRanker-PPO (offline environment + PPO training)



## Module 2 — SentimentBERT (DistilBERT)

Implements sentiment extraction described in the methodology (positive/neutral/negative).

### Train (2-stage recommended)

**Stage 1 (IMDB 2-class warm-start)**
```bash
python -m DeepSentRec.models.sentimentbert.train_sentiment \
  --stage stage1_imdb \
  --train data/processed/imdb_train.parquet \
  --valid data/processed/imdb_test.parquet \
  --output_dir artifacts/sentimentbert
```

**Stage 2 (3-class fine-tune on Amazon/Yelp; rating->3class mapping)**
```bash
python -m DeepSentRec.models.sentimentbert.train_sentiment \
  --stage stage2_finetune \
  --train data/processed/amazon_yelp_train.parquet \
  --valid data/processed/amazon_yelp_valid.parquet \
  --resume_from artifacts/sentimentbert/imdb_2class_best \
  --output_dir artifacts/sentimentbert
```

### Inference (generate sentiment labels + embeddings)
```bash
python -m DeepSentRec.models.sentimentbert.infer_sentiment \
  --model_dir artifacts/sentimentbert/finetune_3class_best \
  --data data/processed/amazon.parquet \
  --out_table data/processed/amazon_with_sentiment.parquet \
  --out_embeddings data/processed/amazon_sentiment_embeds.npy \
  --label_mode 3class
```


## Module 3 — HybridCF-SBERT (NMF + SBERT)

### 3.1 Build interactions
```bash
python -m DeepSentRec.models.hybrid.build_interactions \
  --input data/processed/amazon_with_sentiment.parquet \
  --out_dir artifacts/interactions \
  --min_user_inter 5 --min_item_inter 5
```

### 3.2 Build SBERT item embeddings
```bash
python -m DeepSentRec.models.hybrid.sbert_items \
  --input data/processed/amazon_with_sentiment.parquet \
  --out_dir artifacts/sbert_items \
  --max_reviews_per_item 3
```

### 3.3 Train NMF + user profiles (Hybrid artifacts)
```bash
python -m DeepSentRec.models.hybrid.train_hybrid \
  --interactions artifacts/interactions/interactions.parquet \
  --mappings artifacts/interactions/mappings.npz \
  --item_ids artifacts/sbert_items/item_ids.npy \
  --item_embeddings artifacts/sbert_items/item_embeddings.npy \
  --out_dir artifacts/hybrid \
  --k 50
```

### 3.4 Evaluate (requires a test interactions file)
```bash
python -m DeepSentRec.models.hybrid.evaluate_hybrid \
  --train_interactions artifacts/interactions/interactions.parquet \
  --test_interactions artifacts/interactions/interactions_test.parquet \
  --artifacts artifacts/hybrid \
  --alpha 0.5 --k 10
```


## Module 4 — RLRanker-PPO (Adaptive Reranking)

### 4.1 Build offline episodes (from Hybrid candidates + held-out positives)
```bash
python -m DeepSentRec.models.rl_ranker.build_episodes \
  --train_interactions artifacts/interactions/interactions.parquet \
  --test_interactions artifacts/interactions/interactions_test.parquet \
  --hybrid_artifacts artifacts/hybrid \
  --out artifacts/rl/episodes.jsonl \
  --k_candidates 20 --steps_per_user 10 --max_users 2000
```

### 4.2 Train PPO ranker
```bash
python -m DeepSentRec.models.rl_ranker.train_ppo \
  --episodes artifacts/rl/episodes.jsonl \
  --out_dir artifacts/rl \
  --k_candidates 20 --action_type swap \
  --timesteps 100000 \
  --gamma 0.99 --lr 3e-4 --clip_range 0.2 --batch_size 64
```

### 4.3 Rerank a candidate list (inference utility)
```bash
python -m DeepSentRec.models.rl_ranker.rerank \
  --model artifacts/rl/ppo_ranker.zip \
  --candidates_json '[{"item": 10, "base_score": 0.8, "sentiment_score": 0.6}]' \
  --k_candidates 20 --steps 10
```

> This module provides an offline simulation environment aligned with the methodology's
> “mock-user interactions” training for PPO. For production, connect rewards to live CTR/purchase logs.


## Quickstart (End-to-End)

### 0) Install
```bash
pip install -r requirements.txt
```

### 1) Build standardized tables (Module 1)
Amazon:
```bash
python build_dataset.py --dataset amazon --input data/raw/amazon.json.gz --out data/processed/amazon.parquet --clean --limit 200000
```

Yelp:
```bash
python build_dataset.py --dataset yelp --input data/raw/yelp_review.json --out data/processed/yelp.parquet --clean --limit 200000
```

IMDB (HuggingFace):
```bash
python build_dataset.py --dataset imdb_hf --split train --out data/processed/imdb_train.parquet --limit 50000
python build_dataset.py --dataset imdb_hf --split test  --out data/processed/imdb_test.parquet --limit 50000
```

### 2) Run the full training pipeline (Modules 2–4)
```bash
python main_train_pipeline.py \
  --amazon_table data/processed/amazon.parquet \
  --yelp_table data/processed/yelp.parquet \
  --imdb_train data/processed/imdb_train.parquet \
  --imdb_test data/processed/imdb_test.parquet \
  --out_dir artifacts
```

### 3) Demo API (optional)
```bash
python api/flask_app.py
# then open:
# http://localhost:5000/recommend?user_index=0
```

## Notes
- For very large datasets, run with `--limit` during dataset build and `--limit_items` in SBERT embedding.
- Offline RL is a simulation; for real systems, connect rewards to live CTR/purchase logs.
