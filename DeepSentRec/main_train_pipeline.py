#!/usr/bin/env python
"""
End-to-end training pipeline for DeepSentRec (Modules 1–4).

This script orchestrates:
1) Build standardized datasets (optional; assumes you already ran build_dataset.py)
2) Module 2: Train SentimentBERT (optional if you have a checkpoint)
3) Module 2: Run inference to attach sentiment labels + embeddings (optional)
4) Module 3: Build interactions + split
5) Module 3: Build SBERT item embeddings
6) Module 3: Train HybridCF-SBERT (NMF + user profiles)
7) Module 4: Build RL episodes + Train PPO ranker

Use cases:
- Research reproduction: run step-by-step from CLI
- Automation: run full pipeline when paths are provided

Note: Download datasets separately and put them under data/raw/ or your preferred path.
"""

from __future__ import annotations
import argparse, os, subprocess, sys

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amazon_table", required=True, help="Standardized amazon parquet/csv.")
    ap.add_argument("--yelp_table", required=False, default="", help="Standardized yelp parquet/csv (optional).")
    ap.add_argument("--imdb_train", required=False, default="", help="IMDB train parquet (optional).")
    ap.add_argument("--imdb_test", required=False, default="", help="IMDB test parquet (optional).")
    ap.add_argument("--out_dir", required=True, help="Artifacts output root (e.g., artifacts/).")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k_candidates", type=int, default=20)
    ap.add_argument("--latent_k", type=int, default=50)
    ap.add_argument("--ppo_timesteps", type=int, default=100000)
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    sentiment_dir = os.path.join(out, "sentimentbert")
    sbert_items_dir = os.path.join(out, "sbert_items")
    inter_dir = os.path.join(out, "interactions")
    hybrid_dir = os.path.join(out, "hybrid")
    rl_dir = os.path.join(out, "rl")

    # Optional: IMDB warm-start if provided
    if args.imdb_train and args.imdb_test:
        run([sys.executable, "-m", "DeepSentRec.models.sentimentbert.train_sentiment",
             "--stage","stage1_imdb",
             "--train", args.imdb_train,
             "--valid", args.imdb_test,
             "--output_dir", sentiment_dir])

        resume = os.path.join(sentiment_dir, "imdb_2class_best")
    else:
        resume = ""

    # Build finetune dataset (amazon+yelp)
    finetune_train = args.amazon_table
    finetune_valid = ""  # user can provide if desired
    if args.yelp_table:
        # naive concat for training; user can create their own split
        import pandas as pd
        a = pd.read_parquet(args.amazon_table) if args.amazon_table.endswith(".parquet") else pd.read_csv(args.amazon_table)
        y = pd.read_parquet(args.yelp_table) if args.yelp_table.endswith(".parquet") else pd.read_csv(args.yelp_table)
        comb = pd.concat([a, y], ignore_index=True)
        finetune_train = os.path.join(out, "amazon_yelp_train.parquet")
        comb.to_parquet(finetune_train, index=False)

    # Train 3-class sentiment
    run([sys.executable, "-m", "DeepSentRec.models.sentimentbert.train_sentiment",
         "--stage","stage2_finetune",
         "--train", finetune_train,
         "--valid", finetune_valid,
         "--resume_from", resume,
         "--output_dir", sentiment_dir])

    model_dir = os.path.join(sentiment_dir, "finetune_3class_best")

    # Sentiment inference on amazon table
    amazon_with_sent = os.path.join(out, "amazon_with_sentiment.parquet")
    amazon_embeds = os.path.join(out, "amazon_sentiment_embeds.npy")
    run([sys.executable, "-m", "DeepSentRec.models.sentimentbert.infer_sentiment",
         "--model_dir", model_dir,
         "--data", args.amazon_table,
         "--out_table", amazon_with_sent,
         "--out_embeddings", amazon_embeds,
         "--label_mode", "3class"])

    # Build interactions from amazon_with_sentiment
    run([sys.executable, "-m", "DeepSentRec.models.hybrid.build_interactions",
         "--input", amazon_with_sent,
         "--out_dir", inter_dir,
         "--min_user_inter","5","--min_item_inter","5"])

    # Split interactions
    run([sys.executable, os.path.join("DeepSentRec","data","split_interactions.py"),
         "--input", os.path.join(inter_dir, "interactions.parquet"),
         "--out_dir", inter_dir,
         "--test_n","1","--valid_n","0"])

    # SBERT item embeddings
    run([sys.executable, "-m", "DeepSentRec.models.hybrid.sbert_items",
         "--input", amazon_with_sent,
         "--out_dir", sbert_items_dir,
         "--max_reviews_per_item","3",
         "--limit_items","0"])

    # Train hybrid
    run([sys.executable, "-m", "DeepSentRec.models.hybrid.train_hybrid",
         "--interactions", os.path.join(inter_dir,"interactions_train.parquet"),
         "--mappings", os.path.join(inter_dir,"mappings.npz"),
         "--item_ids", os.path.join(sbert_items_dir,"item_ids.npy"),
         "--item_embeddings", os.path.join(sbert_items_dir,"item_embeddings.npy"),
         "--out_dir", hybrid_dir,
         "--k", str(args.latent_k)])

    # Build RL episodes
    os.makedirs(rl_dir, exist_ok=True)
    run([sys.executable, "-m", "DeepSentRec.models.rl_ranker.build_episodes",
         "--train_interactions", os.path.join(inter_dir,"interactions_train.parquet"),
         "--test_interactions", os.path.join(inter_dir,"interactions_test.parquet"),
         "--hybrid_artifacts", hybrid_dir,
         "--out", os.path.join(rl_dir,"episodes.jsonl"),
         "--k_candidates", str(args.k_candidates),
         "--alpha", str(args.alpha),
         "--steps_per_user","10","--max_users","2000"])

    # Train PPO
    run([sys.executable, "-m", "DeepSentRec.models.rl_ranker.train_ppo",
         "--episodes", os.path.join(rl_dir,"episodes.jsonl"),
         "--out_dir", rl_dir,
         "--k_candidates", str(args.k_candidates),
         "--action_type","swap",
         "--timesteps", str(args.ppo_timesteps)])

    print("\n✅ Training pipeline completed.")
    print("Artifacts:", out)

if __name__ == "__main__":
    main()
