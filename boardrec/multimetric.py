"""Offline ranking evaluation for sparse *observed* ratings.

Evaluate only TRAIN-fitted recommendations against VALIDATION. Unknown/unrated
items are *unlabeled*, not known negatives: observed Recall/NDCG can undervalue
relevant recommendations. Diversity/coverage/novelty do not need heldout labels.

No user identities or recommendation lists are written into the public report.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .content import ContentBasedRecommender
from .preprocessing import load_prepared
from .ranking import PopularityRanker, _percentile_ranks
from .ranking_experiments import eligible_sample


def _per_user_metrics(predictions: list[str], relevant: dict[str, float], k: int = 10):
    """Graded NDCG: rating 8->gain 1; 9->2; 10->3; below 8->0.

    Only observed held-out ratings are labeled; other candidates are unknown.
    """
    gains = [max(0.0, min(3.0, float(relevant.get(str(g), 0.0)) - 7.0))
             for g in predictions[:k]]
    actual = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((max(0.0, min(3.0, float(r) - 7.0)) for r in relevant.values()), reverse=True)[:k]
    best = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    count = sum(g > 0 for g in gains)
    return {"hit": float(count > 0), "recall": count / len(relevant) if relevant else 0.0,
            "ndcg": actual / best if best else 0.0}


def _item_diversity(positions, normalized_features):
    """Mean pairwise 1-cosine using mechanics+themes normalized binary rows."""
    if len(positions) < 2:
        return 0.0
    m = normalized_features[positions]
    sims = (m @ m.T).toarray()
    n = len(positions)
    return float(np.clip(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1)), 0, 1))


def compare_on_validation(prepared_dir: str | Path, *, max_users: int = 1000,
                          k: int = 10, seed: int = 42,
                          alphas=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)) -> pd.DataFrame:
    """Compare popularity and content ranks on the *same sampled validation users*.

    alpha=1 => pure content; alpha=0 => popularity-only baseline.
    Alpha is not automatically selected; interpret accuracy/diversity tradeoffs.
    """
    if k < 1 or max_users < 1:
        raise ValueError("k and max_users must be positive")
    if not alphas or any(not 0 <= a <= 1 for a in alphas):
        raise ValueError("alpha weights must be between 0 and 1")

    train, val, _, games, mechanics, themes = load_prepared(prepared_dir)
    users, _, eligible_count = eligible_sample(val, threshold=8, max_users=max_users, seed=seed)
    if not users:
        raise ValueError("No users with positive validation ratings")
    val_pos = val.loc[val.rating >= 8]
    relevant = {str(uid): dict(zip(group.game_id.astype(str), group.rating.astype(float)))
                for uid, group in val_pos.groupby("user_id")}
    model = ContentBasedRecommender(feature_set="both").fit(train, games, mechanics, themes)
    pop = PopularityRanker().fit(train, model.game_ids_)
    catalog = list(model.game_ids_)
    position = model.game_positions_
    counts = np.array([pop.popularity_[g] for g in catalog], dtype=float)
    catalog_pop_percentile = dict(zip(catalog, _percentile_ranks(counts)))
    total_obs = sum(pop.popularity_.values())
    # Smoothed empirical probability; novel items have larger surprisal.
    item_novelty = {g: -math.log2((pop.popularity_[g] + 1) /
                                (total_obs + len(catalog))) for g in catalog}
    # Long-tail = catalog outside top 20% by TRAIN rating count; no holdout leakage.
    top_count = max(1, math.ceil(0.2 * len(catalog)))
    head = {catalog[i] for i in np.argsort(-counts, kind="stable")[:top_count]}
    tail = set(catalog) - head
    tail_eligible_users = sum(bool(set(rel) & tail) for uid, rel in relevant.items() if uid in users)
    summaries = {float(a): {"hit": 0., "recall": 0., "ndcg": 0., "diversity": 0.,
                            "novelty": 0., "pop_percentile": 0.,
                            "distinct": set(), "exposure_count": Counter(),
                            "tail_hits": 0., "tail_recall": 0., "tail_exposure": 0.}
                 for a in alphas}
    for uid in users:
        candidates, popularity_rank = pop.rank_for_user(uid)
        content_score = model.predict_many(uid, candidates)
        content_rank = _percentile_ranks(content_score)
        actual = relevant[uid]
        for alpha, s in summaries.items():
            combined = alpha * content_rank + (1 - alpha) * popularity_rank
            best_indices = np.argsort(-combined, kind="stable")[:k]
            chosen = [candidates[i] for i in best_indices]
            stats = _per_user_metrics(chosen, actual, k=k)
            for key in ("hit", "recall", "ndcg"):
                s[key] += stats[key]
            s["distinct"].update(chosen)
            s["exposure_count"].update(chosen)
            tail_relevant = set(actual) & tail
            if tail_relevant:
                caught = len(set(chosen) & tail_relevant)
                s["tail_hits"] += bool(caught)
                s["tail_recall"] += caught / len(tail_relevant)
            s["tail_exposure"] += sum(g in tail for g in chosen) / k
            s["diversity"] += _item_diversity([position[g] for g in chosen], model.features_)
            s["novelty"] += float(np.mean([item_novelty[g] for g in chosen]))
            s["pop_percentile"] += float(np.mean([catalog_pop_percentile[g] for g in chosen]))

    rows = []
    n = len(users)
    for alpha, s in summaries.items():
        pairs = n * (n - 1) / 2
        exposure_pairs = sum(c * (c - 1) / 2 for c in s["exposure_count"].values())
        avg_overlap_fraction = exposure_pairs / (pairs * k) if pairs else 0.0
        rows.append({"content_weight": alpha, "popularity_weight": 1-alpha,
                     "hit_rate_at_10": s["hit"] / n,
                     "observed_recall_at_10": s["recall"] / n,
                     "observed_ndcg_at_10": s["ndcg"] / n,
                     "catalog_coverage_at_10": len(s["distinct"]) / len(catalog),
                     "distinct_recommended_games": len(s["distinct"]),
                     "intra_list_diversity_at_10": s["diversity"] / n,
                     "inter_user_personalization_at_10": 1 - avg_overlap_fraction,
                     "mean_recommendation_novelty_bits": s["novelty"] / n,
                     "mean_recommended_popularity_percentile": s["pop_percentile"] / n,
                     "tail_hit_rate_at_10": s["tail_hits"] / tail_eligible_users if tail_eligible_users else float("nan"),
                     "tail_observed_recall_at_10": s["tail_recall"] / tail_eligible_users if tail_eligible_users else float("nan"),
                     "tail_recommendation_fraction": s["tail_exposure"] / n,
                     "tail_eligible_users": tail_eligible_users,
                     "evaluated_users": n, "eligible_users": int(eligible_count),
                     "candidate_catalog_size": len(catalog), "k": k})
    return pd.DataFrame(rows).sort_values("content_weight").reset_index(drop=True)


def run_multimetric(prepared_dir: str | Path, output_dir: str | Path, **kwargs):
    results = compare_on_validation(prepared_dir, **kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "validation_multi_metric.csv", index=False, float_format="%.6f")
    meta = {"data": "User-provided BGG dataset, preprocessed sample",
            "split": "Validation only. Test not used for metric/weight selection.",
            "selection": "No single weight automatically selected; compare competing objectives.",
            "labels": "Positive when observed held-out rating >= 8; unobserved means unknown, NOT negative.",
            "ndcg_gain": "max(0, min(3, rating - 7)) for observed held-out ratings; log2 rank discount.",
            "observed_recall": "Per-user fraction of held-out positive ratings present in top-k.",
            "coverage": "Unique games in all top-k lists / 1000-game evaluated catalog.",
            "diversity": "Average pairwise 1 - cosine(mechanics+themes) within each top-k list.",
            "personalization": "1 - mean overlap / k of two users' top-k recommendation sets.",
            "novelty": "Average -log2 Laplace-smoothed TRAIN item popularity probability (bits).",
            "long_tail": "Within the already popularity-filtered 1000-game catalog, bottom 80% by TRAIN rating count. Tail Hit/Recall evaluated on users with heldout positive tail items; count disclosed.",
            "caveats": ["Offline missing ratings are unlabeled, not negative.",
                        "The candidate catalog already contains 1000 globally popular games.",
                        "The TEST split was used in a previous experiment. Future comparisons require a fresh untouched holdout.",
                        "Only aggregated output saved; no raw usernames or rating records are published."]}
    (out / "metrics_definitions.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return results
