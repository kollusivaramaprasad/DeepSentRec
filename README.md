# DeepSentRec

## Deep Learning-Based Sentiment-Aware Product Recommendation System with Hybrid Filtering and Reinforcement Learning

------------------------------------------------------------------------

## 📌 Overview

DeepSentRec is a research-grade Sentiment-Aware Hybrid Recommendation
System enhanced with Reinforcement Learning-based dynamic reranking.

It integrates: - Transformer-based Sentiment Modeling (DistilBERT) -
Hybrid Collaborative + Content Filtering (NMF + SBERT) - PPO-based
Reinforcement Learning Reranking - Full Evaluation Pipeline

------------------------------------------------------------------------

## 🏗 System Architecture

    Raw Reviews
         ↓
    Data Preprocessing
         ↓
    SentimentBERT (DistilBERT)
         ↓
    HybridCF-SBERT
         ├── NMF Collaborative Filtering
         ├── SBERT Item Embeddings
         └── Hybrid Score Fusion
         ↓
    RLRanker-PPO
         ├── Offline Environment
         ├── Reward Modeling
         └── Dynamic Reranking
         ↓
    Final Top-N Recommendations

------------------------------------------------------------------------

## 📂 Repository Structure

    DeepSentRec/
    │
    ├── data/
    │   ├── amazon_loader.py
    │   ├── yelp_loader.py
    │   ├── kaggle_loader.py
    │   ├── imdb_loader.py
    │   └── split_interactions.py
    │
    ├── preprocessing/
    │   └── text_cleaner.py
    │
    ├── models/
    │   ├── sentimentbert/
    │   │   ├── train_sentiment.py
    │   │   ├── infer_sentiment.py
    │   │   └── dataset.py
    │   │
    │   ├── hybrid/
    │   │   ├── build_interactions.py
    │   │   ├── nmf_model.py
    │   │   ├── sbert_items.py
    │   │   ├── hybrid_scorer.py
    │   │   ├── train_hybrid.py
    │   │   └── evaluate_hybrid.py
    │   │
    │   └── rl_ranker/
    │       ├── environment.py
    │       ├── reward_function.py
    │       ├── build_episodes.py
    │       ├── train_ppo.py
    │       └── rerank.py
    │
    ├── evaluation/
    │   └── recommendation_metrics.py
    │
    ├── api/
    │   └── flask_app.py
    │
    ├── configs/
    │   └── hyperparameters.yaml
    │
    ├── build_dataset.py
    ├── main_train_pipeline.py
    ├── main_inference_pipeline.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔧 Installation

### Clone Repository

``` bash
git clone https://github.com/YOUR_USERNAME/DeepSentRec.git
cd DeepSentRec
```

### Create Environment

``` bash
conda create -n deepsentrec python=3.10
conda activate deepsentrec
```

or

``` bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📊 Datasets

Place datasets inside:

    data/raw/

Supported datasets: - Amazon Reviews - Yelp Dataset - IMDB Reviews -
Kaggle Reviews

------------------------------------------------------------------------

## 🚀 Usage

### Dataset Preparation

``` bash
python build_dataset.py --dataset amazon \
--input data/raw/amazon.json.gz \
--out data/processed/amazon.parquet \
--clean
```

### Train Sentiment Model

``` bash
python -m DeepSentRec.models.sentimentbert.train_sentiment \
--stage stage1_imdb
```

### Train Hybrid Model

``` bash
python -m DeepSentRec.models.hybrid.train_hybrid
```

### Train PPO Ranker

``` bash
python -m DeepSentRec.models.rl_ranker.train_ppo \
--timesteps 100000
```

------------------------------------------------------------------------

## 🌐 API Demo

``` bash
python api/flask_app.py
```

Open:

    http://localhost:5000/recommend?user_index=0

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   Precision@K
-   Recall@K
-   HitRate@K
-   NDCG@K
-   MAP@K

------------------------------------------------------------------------

## ⚙️ Hyperparameters

Modify:

    configs/hyperparameters.yaml

------------------------------------------------------------------------

## 📜 License

Released for research and academic purposes.

------------------------------------------------------------------------


