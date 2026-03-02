"""
Evaluation Metrics for the Restaurant Recommendation Engine.

Metrics implemented:
  Accuracy:
    - Precision@K
    - Recall@K
    - NDCG@K  (Normalized Discounted Cumulative Gain)

  Diversity:
    - Intra-List Diversity (ILD) — average pairwise embedding distance
    - Category Coverage — fraction of distinct cuisines in recommended list

  Popularity:
    - Average rating of recommended items

All metrics accept a list of recommended item IDs and a ground-truth set.
"""
from __future__ import annotations  # makes all annotations strings → works on Python 3.9+

import math
from typing import Callable, Optional

import numpy as np


# ─── Accuracy Metrics ──────────────────────────────────────────────────────────

def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """Fraction of relevant items found in the top-K recommendations."""
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain at K.
    Relevance is binary: 1 if item is in relevant_ids, else 0.
    """
    if not relevant_ids or k == 0:
        return 0.0

    def dcg(ids: list, rel: set, k: int) -> float:
        return sum(
            (1.0 / math.log2(i + 2))
            for i, rid in enumerate(ids[:k])
            if rid in rel
        )

    actual_dcg = dcg(recommended_ids, relevant_ids, k)
    # Ideal: place all relevant items first
    ideal_list = list(relevant_ids)[:k]
    ideal_dcg = dcg(ideal_list, relevant_ids, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ─── Diversity Metrics ─────────────────────────────────────────────────────────

def intra_list_diversity(
    recommended_ids: list,
    embeddings: np.ndarray,
    id_to_idx: dict,
) -> float:
    """
    Intra-List Diversity (ILD): average pairwise cosine DISTANCE (1 - similarity)
    among recommended items.  Higher = more diverse.
    """
    idxs = [id_to_idx[rid] for rid in recommended_ids if rid in id_to_idx]
    if len(idxs) < 2:
        return 0.0

    vecs = embeddings[idxs]          # (n, d)
    sim_matrix = vecs @ vecs.T       # cosine sims (embeddings are L2-normalised)
    n = len(idxs)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sim_matrix[i, j])
            count += 1
    return total / count if count else 0.0


def category_coverage(recommended: list, all_restaurants: list) -> float:
    """
    Fraction of distinct cuisines in the database that appear in
    the recommended list.
    """
    all_cuisines = set(r["cuisine"] for r in all_restaurants)
    rec_cuisines = set(r["cuisine"] for r in recommended)
    return len(rec_cuisines) / len(all_cuisines) if all_cuisines else 0.0


# ─── User Satisfaction Proxy ───────────────────────────────────────────────────

def average_rating(recommended: list) -> float:
    """Average star rating of recommended items — proxy for quality."""
    if not recommended:
        return 0.0
    return sum(r["rating"] for r in recommended) / len(recommended)


def like_ratio(feedback: dict) -> float:
    """
    Fraction of items with a 'like' signal.
    feedback = {restaurant_id: 'like' | 'dislike'}
    """
    if not feedback:
        return 0.0
    likes = sum(1 for v in feedback.values() if v == "like")
    return likes / len(feedback)


# ─── Simulated Evaluation ─────────────────────────────────────────────────────

def simulate_evaluation(
    restaurants: list,
    recommender_fn: Callable,
    n_users: int = 50,
    k: int = 10,
    seed: int = 42,
    embeddings: Optional[np.ndarray] = None,
    id_to_idx: Optional[dict] = None,
) -> dict:
    """
    Simulate evaluation across synthetic user profiles.

    Each synthetic user:
    - "likes" 3-5 restaurants from a cuisine they prefer
    - Ground truth = other restaurants of same cuisine
    - We evaluate how many ground-truth items appear in top-K recommendations

    Parameters
    ----------
    embeddings : pre-computed embedding matrix (passed in from app to avoid
                 circular import). If None, ILD metric is skipped.
    id_to_idx  : mapping from restaurant id to row index in embeddings.
    """
    import random
    rng = random.Random(seed)

    cuisines = list(set(r["cuisine"] for r in restaurants))

    metrics_accumulator: dict = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "ild": [],
        "category_coverage": [],
        "avg_rating": [],
    }

    for _ in range(n_users):
        preferred_cuisine = rng.choice(cuisines)

        cuisine_restaurants = [r for r in restaurants if r["cuisine"] == preferred_cuisine]
        if len(cuisine_restaurants) < 6:
            continue

        # Sample liked items (history)
        liked = rng.sample(cuisine_restaurants, min(4, len(cuisine_restaurants) // 2))
        liked_ids = [r["id"] for r in liked]
        liked_set = set(liked_ids)

        # Ground truth = remaining same-cuisine restaurants
        ground_truth = {r["id"] for r in cuisine_restaurants if r["id"] not in liked_set}
        if not ground_truth:
            continue

        # Get recommendations
        try:
            recs = recommender_fn(
                restaurants=restaurants,
                liked_ids=liked_ids,
                exclude_ids=liked_set,
                top_k=k,
            )
        except Exception:
            continue

        rec_ids = [r["id"] for r in recs]

        # Accuracy metrics
        metrics_accumulator["precision"].append(precision_at_k(rec_ids, ground_truth, k))
        metrics_accumulator["recall"].append(recall_at_k(rec_ids, ground_truth, k))
        metrics_accumulator["ndcg"].append(ndcg_at_k(rec_ids, ground_truth, k))

        # Diversity metrics
        if embeddings is not None and id_to_idx is not None and rec_ids:
            metrics_accumulator["ild"].append(
                intra_list_diversity(rec_ids, embeddings, id_to_idx)
            )
        metrics_accumulator["category_coverage"].append(category_coverage(recs, restaurants))
        metrics_accumulator["avg_rating"].append(average_rating(recs))

    def safe_mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    return {
        f"Precision@{k}":         safe_mean(metrics_accumulator["precision"]),
        f"Recall@{k}":            safe_mean(metrics_accumulator["recall"]),
        f"NDCG@{k}":              safe_mean(metrics_accumulator["ndcg"]),
        "Intra-List Diversity":   safe_mean(metrics_accumulator["ild"]),
        "Category Coverage":      safe_mean(metrics_accumulator["category_coverage"]),
        "Avg Recommended Rating": safe_mean(metrics_accumulator["avg_rating"]),
        "N Users Evaluated":      len(metrics_accumulator["precision"]),
    }
