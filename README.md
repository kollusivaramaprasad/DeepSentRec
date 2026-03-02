________________________________________
DeepSentRec
A Deep Learning-Based Sentiment-Aware Product Recommendation System with Hybrid Filtering and Reinforcement Learning
________________________________________
📌 Overview
DeepSentRec is a complete, research-grade implementation of a Sentiment-Aware Hybrid Recommendation System enhanced with Reinforcement Learning-based dynamic reranking.
The framework integrates:
•	🔎 Transformer-based Sentiment Modeling (DistilBERT)
•	🤝 Hybrid Collaborative + Content-Based Filtering (NMF + SBERT)
•	🎯 Adaptive Reranking using PPO (Proximal Policy Optimization)
•	📊 Full evaluation pipeline with Precision@K, NDCG@K, MAP@K, HitRate
The implementation strictly follows the proposed methodology of the DeepSentRec research framework and is modularized into four primary components.
________________________________________
🏗 System Architecture
Raw Reviews
   ↓
Module 1: Data Standardization & Preprocessing
   ↓
Module 2: SentimentBERT (DistilBERT Fine-Tuning)
   ↓
Module 3: HybridCF-SBERT
        - NMF Collaborative Filtering
        - SBERT Item Embeddings
        - Hybrid Score Fusion
   ↓
Module 4: RLRanker-PPO
        - Offline RL Environment
        - Reward Modeling
        - Dynamic Reranking
   ↓
Final Top-N Recommendations
________________________________________
📂 Repository Structure
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
________________________________________
🔧 Installation
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/DeepSentRec.git
cd DeepSentRec
2️⃣ Create environment (recommended)
conda create -n deepsentrec python=3.10
conda activate deepsentrec
or
python -m venv venv
source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
Key libraries:
•	PyTorch
•	Transformers
•	Sentence-Transformers
•	scikit-learn
•	Stable-Baselines3
•	Gymnasium
•	Flask
•	Pandas / NumPy
________________________________________
📊 Datasets
You must download datasets separately and place them under data/raw/.
Supported Datasets
Dataset	Purpose	Source
Amazon Reviews	Recommendation training	UCSD McAuley Lab
Yelp Open Dataset	Cross-domain training	Yelp Dataset
IMDB Reviews	Sentiment pretraining	HuggingFace imdb
Kaggle E-Commerce Reviews	Optional evaluation	Kaggle
________________________________________
🚀 Usage Guide
________________________________________
MODULE 1 — Dataset Standardization
Amazon
python build_dataset.py \
  --dataset amazon \
  --input data/raw/amazon.json.gz \
  --out data/processed/amazon.parquet \
  --clean \
  --limit 200000
Yelp
python build_dataset.py \
  --dataset yelp \
  --input data/raw/yelp_review.json \
  --out data/processed/yelp.parquet \
  --clean
IMDB
python build_dataset.py \
  --dataset imdb_hf \
  --split train \
  --out data/processed/imdb_train.parquet
________________________________________
MODULE 2 — SentimentBERT
Stage 1: IMDB Pretraining
python -m DeepSentRec.models.sentimentbert.train_sentiment \
  --stage stage1_imdb \
  --train data/processed/imdb_train.parquet \
  --valid data/processed/imdb_test.parquet \
  --output_dir artifacts/sentimentbert
Stage 2: Fine-tune on Amazon/Yelp
python -m DeepSentRec.models.sentimentbert.train_sentiment \
  --stage stage2_finetune \
  --train data/processed/amazon.parquet \
  --output_dir artifacts/sentimentbert
Inference
python -m DeepSentRec.models.sentimentbert.infer_sentiment \
  --model_dir artifacts/sentimentbert/finetune_3class_best \
  --data data/processed/amazon.parquet \
  --out_table data/processed/amazon_with_sentiment.parquet \
  --out_embeddings data/processed/amazon_sentiment_embeds.npy
________________________________________
MODULE 3 — HybridCF-SBERT
Build Interactions
python -m DeepSentRec.models.hybrid.build_interactions \
  --input data/processed/amazon_with_sentiment.parquet \
  --out_dir artifacts/interactions
Split Train/Test
python data/split_interactions.py \
  --input artifacts/interactions/interactions.parquet \
  --out_dir artifacts/interactions
Build SBERT Embeddings
python -m DeepSentRec.models.hybrid.sbert_items \
  --input data/processed/amazon_with_sentiment.parquet \
  --out_dir artifacts/sbert_items
Train Hybrid Model
python -m DeepSentRec.models.hybrid.train_hybrid \
  --interactions artifacts/interactions/interactions_train.parquet \
  --mappings artifacts/interactions/mappings.npz \
  --item_ids artifacts/sbert_items/item_ids.npy \
  --item_embeddings artifacts/sbert_items/item_embeddings.npy \
  --out_dir artifacts/hybrid
Evaluate
python -m DeepSentRec.models.hybrid.evaluate_hybrid \
  --train_interactions artifacts/interactions/interactions_train.parquet \
  --test_interactions artifacts/interactions/interactions_test.parquet \
  --artifacts artifacts/hybrid
________________________________________
MODULE 4 — RLRanker-PPO
Build Offline Episodes
python -m DeepSentRec.models.rl_ranker.build_episodes \
  --train_interactions artifacts/interactions/interactions_train.parquet \
  --test_interactions artifacts/interactions/interactions_test.parquet \
  --hybrid_artifacts artifacts/hybrid \
  --out artifacts/rl/episodes.jsonl
Train PPO
python -m DeepSentRec.models.rl_ranker.train_ppo \
  --episodes artifacts/rl/episodes.jsonl \
  --out_dir artifacts/rl \
  --timesteps 100000
________________________________________
🧠 End-to-End Training
python main_train_pipeline.py \
  --amazon_table data/processed/amazon.parquet \
  --imdb_train data/processed/imdb_train.parquet \
  --imdb_test data/processed/imdb_test.parquet \
  --out_dir artifacts
________________________________________
🌐 API Demo
python api/flask_app.py
Access:
http://localhost:5000/recommend?user_index=0
________________________________________
📈 Evaluation Metrics
•	Precision@K
•	Recall@K
•	HitRate@K
•	NDCG@K
•	MAP@K
Located in:
evaluation/recommendation_metrics.py
________________________________________
⚙️ Hyperparameters
Edit:
configs/hyperparameters.yaml
Includes:
•	Sentiment model learning rate
•	NMF latent factors
•	Hybrid α weight
•	PPO gamma, clip range
•	Reward weights
________________________________________
🔬 Research Notes
•	DistilBERT fine-tuned for 3-class sentiment
•	NMF latent representation for collaborative filtering
•	SBERT for semantic similarity
•	PPO for adaptive ranking optimization
•	Offline RL simulation based on interaction logs
________________________________________
⚠️ Important Notes
•	RL component is offline simulation.
•	For production, connect reward function to live click logs.
•	Large datasets require GPU for transformer training.
•	SBERT embedding generation can be memory intensive.
________________________________________
📜 License
This repository is released for research and academic purposes.
________________________________________
📧 Contact
For research collaboration or questions:
•	Author: [Your Name]
•	Email: your.email@example.com
•	Institution: [Your Institution]
________________________________________

