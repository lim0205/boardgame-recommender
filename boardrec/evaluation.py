"""Shared offline evaluation for any recommender implementing the same API.

Do not train or tune with the test split. RMSE/MAE include all held-out ratings.
HitRate@10 measures whether ANY held-out positive item (rating >= threshold)
was recommended among unobserved catalog items. Macro Recall@10 averages
per-user relevant-item retrieval on the SAME sampled eligible users.
Catalog Coverage@10 measures recommendation breadth on those users.
Unrated != negative; Hit Rate is an incomplete-label offline proxy.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def rating_metrics(model, holdout: pd.DataFrame) -> dict:
    if holdout.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "n_ratings": 0}
    total_squared, total_abs, n = 0.0, 0.0, 0
    # RMSE and MAE use every observed held-out rating, never only Top-10 hits.
    for uid, group in holdout.groupby("user_id", sort=False):
        pred = np.asarray(model.predict_many(str(uid), group.game_id.astype(str).tolist()), dtype=float)
        target = group.rating.to_numpy(dtype=float)
        if len(pred) != len(target) or not np.isfinite(pred).all():
            raise ValueError("Predictor returned invalid shape or non-finite value")
        # err @ err sums squared errors; sqrt(sum / n) gives RMSE in rating points.
        err = pred - target
        total_squared += float(err @ err)
        total_abs += float(np.abs(err).sum())
        n += len(err)
    return {"rmse": math.sqrt(total_squared / n), "mae": total_abs / n, "n_ratings": n}


def hit_rate_at_k(model, holdout: pd.DataFrame, k: int = 10,
                  relevant_threshold: float = 8.0, max_users: int | None = None) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    # A relevant item means an OBSERVED held-out rating >= 8 (default).
    positive = holdout.loc[holdout.rating >= relevant_threshold]
    # Eligible users have at least one relevant item in the held-out data.
    positive_users = sorted(positive.user_id.astype(str).unique())
    eligible_total = len(positive_users)
    if max_users is not None:
        if max_users < 1:
            raise ValueError("max_users must be positive")
        # Deterministic sample avoids bias from lexicographically first users.
        if len(positive_users) > max_users:
            positive_users = sorted(np.random.default_rng(42).choice(
                positive_users, max_users, replace=False).tolist())
    positives_by_user = {
        str(uid): set(group.game_id.astype(str))
        for uid, group in positive.groupby("user_id")
    }
    hits = 0
    recall_sum = 0.0
    relevant_retrieved = 0
    relevant_total = 0
    distinct_recommendations: set[str] = set()
    # Compute Top-10 metrics on the same seeded sample of eligible users.
    for uid in positive_users:
        recommended = set(model.recommend(uid, n=k).game_id.astype(str))
        # Hit is binary per user, whereas Recall counts ALL observed positives recovered.
        # Both use the same ranked list and only positive Test ratings, never unknown labels.
        relevant_items = positives_by_user[uid]
        retrieved = len(recommended & relevant_items)
        hits += bool(retrieved)
        recall_sum += retrieved / len(relevant_items)
        relevant_retrieved += retrieved
        relevant_total += len(relevant_items)
        distinct_recommendations.update(recommended)
    n_eval = len(positive_users)

    # Coverage is comparable only if every model shares the same catalog.
    # ContentBasedRecommender exposes game_ids_; PredictAdapter exposes catalog_.
    catalog = getattr(model, "game_ids_", None)
    if catalog is None:
        catalog = getattr(model, "catalog_", None)
    catalog_size = len(catalog) if catalog is not None else None
    # Coverage measures distinct recommended catalog items, not Recall@10.
    coverage = (len(distinct_recommendations) / catalog_size
                if n_eval and catalog_size else float("nan"))
    return {f"hit_rate@{k}": hits / n_eval if n_eval else float("nan"),
            # Macro-average: each eligible sampled user contributes equally.
            f"recall@{k}": recall_sum / n_eval if n_eval else float("nan"),
            f"catalog_coverage@{k}": coverage,
            "distinct_recommended_games": len(distinct_recommendations),
            "candidate_catalog_size": catalog_size,
            "eligible_users": eligible_total, "evaluated_users": n_eval,
            "hit_count": int(hits), "relevant_retrieved": int(relevant_retrieved),
            "relevant_total": int(relevant_total),
            "relevance_threshold": relevant_threshold,
            "limitation": "Sparse held-out positives; unknown games are not negatives."}


def evaluate(model, holdout: pd.DataFrame, *, top_k: int = 10,
             threshold: float = 8.0, max_hit_users: int | None = 500) -> dict:
    result = rating_metrics(model, holdout)
    result.update(hit_rate_at_k(model, holdout, k=top_k,
                                relevant_threshold=threshold, max_users=max_hit_users))
    return result
