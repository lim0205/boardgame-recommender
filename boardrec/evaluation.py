"""Shared offline evaluation for any recommender implementing the same API.

Do not train or tune with the test split. RMSE/MAE include all held-out ratings.
HitRate@10 measures whether ANY held-out positive item (rating >= threshold)
was recommended among all unobserved catalog items. Unrated != negative;
this is an incomplete-label offline proxy, not an online success rate.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def rating_metrics(model, holdout: pd.DataFrame) -> dict:
    if holdout.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "n_ratings": 0}
    total_squared, total_abs, n = 0.0, 0.0, 0
    for uid, group in holdout.groupby("user_id", sort=False):
        pred = np.asarray(model.predict_many(str(uid), group.game_id.astype(str).tolist()), dtype=float)
        target = group.rating.to_numpy(dtype=float)
        if len(pred) != len(target) or not np.isfinite(pred).all():
            raise ValueError("Predictor returned invalid shape or non-finite value")
        err = pred - target
        total_squared += float(err @ err)
        total_abs += float(np.abs(err).sum())
        n += len(err)
    return {"rmse": math.sqrt(total_squared / n), "mae": total_abs / n, "n_ratings": n}


def hit_rate_at_k(model, holdout: pd.DataFrame, k: int = 10,
                  relevant_threshold: float = 8.0, max_users: int | None = None) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
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
    hits = sum(bool(set(model.recommend(uid, n=k).game_id.astype(str)) & positives_by_user[uid])
               for uid in positive_users)
    n_eval = len(positive_users)
    return {f"hit_rate@{k}": hits / n_eval if n_eval else float("nan"),
            "eligible_users": eligible_total, "evaluated_users": n_eval,
            "hit_count": int(hits), "relevance_threshold": relevant_threshold,
            "limitation": "Sparse held-out positives; unknown games are not negatives."}


def evaluate(model, holdout: pd.DataFrame, *, top_k: int = 10,
             threshold: float = 8.0, max_hit_users: int | None = 500) -> dict:
    result = rating_metrics(model, holdout)
    result.update(hit_rate_at_k(model, holdout, k=top_k,
                                relevant_threshold=threshold, max_users=max_hit_users))
    return result
