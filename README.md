# DeepSentRec
## A Deep Learning-Based Sentiment-Aware Product Recommendation System with Hybrid Filtering and Reinforcement Learning

---

## 📌 Overview

**DeepSentRec** is a research-grade implementation of a **Sentiment-Aware Hybrid Recommendation System** enhanced with **Reinforcement Learning-based Dynamic Reranking**.

The framework combines Natural Language Processing, Recommendation Systems, and Reinforcement Learning to provide adaptive and personalized Top-N product recommendations.

---

## 🚀 Key Features

- 🔎 Transformer-based Sentiment Modeling using **DistilBERT**
- 🤝 Hybrid Recommendation Framework
  - Collaborative Filtering (NMF)
  - Content-Based Filtering (SBERT)
- 🎯 Reinforcement Learning Reranking using **PPO**
- 📊 Complete Recommendation Evaluation Pipeline
- 🌐 REST API Deployment using Flask
- 🧠 Modular Research-Oriented Architecture

---

## 🏗 System Architecture


Raw Reviews
↓
Data Standardization & Preprocessing
↓
SentimentBERT (DistilBERT Fine-Tuning)
↓
HybridCF-SBERT
├── NMF Collaborative Filtering
├── SBERT Item Embeddings
└── Hybrid Score Fusion
↓
RLRanker-PPO
├── Offline RL Environment
├── Reward Modeling
└── Dynamic Reranking
↓
Final Top-N Recommendations


---

## 📂 Repository Structure


DeepSentRec/
│
├── data/
│ ├── amazon_loader.py
│ ├── yelp_loader.py
│ ├── kaggle_loader.py
│ ├── imdb_loader.py
│ └── split_interactions.py
│
├── preprocessing/
│ └── text_cleaner.py
│
├── models/
│ ├── sentimentbert/
│ │ ├── train_sentiment.py
│ │ ├── infer_sentiment.py
│ │ └── dataset.py
│ │
│ ├── hybrid/
│ │ ├── build_interactions.py
│ │ ├── nmf_model.py
│ │ ├── sbert_items.py
│ │ ├── hybrid_scorer.py
│ │ ├── train_hybrid.py
│ │ └── evaluate_hybrid.py
│ │
│ └── rl_ranker/
│ ├── environment.py
│ ├── reward_function.py
│ ├── build_episodes.py
│ ├── train_ppo.py
│ └── rerank.py
│
├── evaluation/
│ └── recommendation_metrics.py
│
├── api/
│ └── flask_app.py
│
├── configs/
│ └── hyperparameters.yaml
│
├── build_dataset.py
├── main_train_pipeline.py
├── main_inference_pipeline.py
├── requirements.txt
└── README.md


---

## 🔧 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepSentRec.git
cd DeepSentRec
2️⃣ Create Environment
Conda (Recommended)
conda create -n deepsentrec python=3.10
conda activate deepsentrec
Virtual Environment
python -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
📦 Main Libraries Used

PyTorch

Transformers

Sentence-Transformers

Scikit-learn

Stable-Baselines3

Gymnasium

Flask

Pandas

NumPy

📊 Datasets

Download datasets manually and place them in:

data/raw/
Dataset	Purpose	Source
Amazon Reviews	Recommendation Training	UCSD McAuley Lab
Yelp Dataset	Cross-Domain Training	Yelp
IMDB Reviews	Sentiment Pretraining	HuggingFace
Kaggle Reviews	Optional Evaluation	Kaggle
🚀 Usage Guide
MODULE 1 — Dataset Standardization
Amazon Dataset
python build_dataset.py \
  --dataset amazon \
  --input data/raw/amazon.json.gz \
  --out data/processed/amazon.parquet \
  --clean \
  --limit 200000
Yelp Dataset
python build_dataset.py \
  --dataset yelp \
  --input data/raw/yelp_review.json \
  --out data/processed/yelp.parquet \
  --clean
IMDB Dataset
python build_dataset.py \
  --dataset imdb_hf \
  --split train \
  --out data/processed/imdb_train.parquet
MODULE 2 — SentimentBERT
Stage 1: IMDB Pretraining
python -m DeepSentRec.models.sentimentbert.train_sentiment \
  --stage stage1_imdb \
  --train data/processed/imdb_train.parquet \
  --valid data/processed/imdb_test.parquet \
  --output_dir artifacts/sentimentbert
Stage 2: Fine-Tuning
python -m DeepSentRec.models.sentimentbert.train_sentiment \
  --stage stage2_finetune \
  --train data/processed/amazon.parquet \
  --output_dir artifacts/sentimentbert
Sentiment Inference
python -m DeepSentRec.models.sentimentbert.infer_sentiment \
  --model_dir artifacts/sentimentbert \
  --data data/processed/amazon.parquet \
  --out_table data/processed/amazon_with_sentiment.parquet
MODULE 3 — HybridCF-SBERT
Build Interactions
python -m DeepSentRec.models.hybrid.build_interactions
Train Hybrid Model
python -m DeepSentRec.models.hybrid.train_hybrid
Evaluate Hybrid Model
python -m DeepSentRec.models.hybrid.evaluate_hybrid
MODULE 4 — RLRanker (PPO)
Build Offline Episodes
python -m DeepSentRec.models.rl_ranker.build_episodes
Train PPO Ranker
python -m DeepSentRec.models.rl_ranker.train_ppo \
  --timesteps 100000
🧠 End-to-End Training
python main_train_pipeline.py \
  --amazon_table data/processed/amazon.parquet \
  --imdb_train data/processed/imdb_train.parquet \
  --imdb_test data/processed/imdb_test.parquet \
  --out_dir artifacts
🌐 API Demo

Run API:

python api/flask_app.py

Access:

http://localhost:5000/recommend?user_index=0
📈 Evaluation Metrics

Precision@K

Recall@K

HitRate@K

NDCG@K

MAP@K

Location:

evaluation/recommendation_metrics.py
⚙️ Hyperparameter Configuration

Edit:

configs/hyperparameters.yaml

Includes:

Sentiment Learning Rate

NMF Latent Factors

Hybrid Fusion Weight

PPO Gamma

PPO Clip Range

Reward Weights

🔬 Research Highlights

DistilBERT-based Sentiment Understanding

Hybrid Collaborative + Content Filtering

Semantic Similarity via SBERT

PPO-based Adaptive Recommendation Ranking

Offline Reinforcement Learning Simulation

⚠️ Important Notes

RL component operates in offline simulation mode.

GPU recommended for transformer training.

Large datasets require high memory.

Production deployment requires live interaction feedback.

📜 License

This project is released for research and academic purposes only.
