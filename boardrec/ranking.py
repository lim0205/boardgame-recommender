"""Popularity baseline and content/popularity ranking for explicit BGG ratings.

This module ONLY changes the order of recommendations, not the 1--10 rating
predictions. Popularity counts are derived exclusively from TRAIN. A candidate
that a user rated in TRAIN is never recommended. Do not tune on TEST.

Population-based recommendation is a useful baseline but not personalization;
report popularity bias and distinct recommended item coverage separately.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0,1]; ties receive the same value."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    if len(values) < 2 or np.ptp(values) < 1e-12:
        return np.full(len(values), 0.5, dtype=float)
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


@dataclass
class PopularityRanker:
    """Train-count popularity baseline, using the same candidate catalog as content."""

    def fit(self, train: pd.DataFrame, catalog_game_ids):
        needed = {"user_id", "game_id", "rating"}
        if not needed.issubset(train.columns):
            raise ValueError(f"Missing train columns: {needed - set(train.columns)}")
        self.catalog_ = sorted({str(g) for g in catalog_game_ids})
        counts = train.game_id.astype(str).value_counts()
        self.popularity_ = {g: int(counts.get(g, 0)) for g in self.catalog_}
        self.observed_ = {
            str(user): set(df.game_id.astype(str))
            for user, df in train.groupby("user_id", sort=False)
        }
        return self

    def candidates(self, user_id: str):
        previously_rated = self.observed_.get(str(user_id), set())
        return [g for g in self.catalog_ if g not in previously_rated]

    def rank_for_user(self, user_id: str):
        games = self.candidates(user_id)
        counts = np.array([self.popularity_[g] for g in games], dtype=float)
        return games, _percentile_ranks(np.log1p(counts))

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n must be >=1")
        games = self.candidates(user_id)
        values = np.array([self.popularity_[g] for g in games], dtype=float)
        order = np.argsort(-values, kind="stable")[:n]
        return pd.DataFrame({"game_id": [games[i] for i in order],
                             "score": values[order]})


@dataclass
class BlendedRanker:
    """Use rank percentiles to blend preference and popularity without unit mismatch.

    content_weight=1 -> pure Content-based ordering; 0 -> popularity baseline.
    predict/predict_many are kept as the original content rating predictions,
    because blended rank percentiles are NOT predicted ratings.
    """
    content_model: object
    popularity: PopularityRanker
    content_weight: float = 0.5

    def __post_init__(self):
        if not 0 <= self.content_weight <= 1:
            raise ValueError("content_weight must be in [0,1]")

    def predict(self, user_id: str, game_id: str) -> float:
        return self.content_model.predict(user_id, game_id)

    def predict_many(self, user_id: str, game_ids):
        return self.content_model.predict_many(user_id, game_ids)

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n must be >=1")
        game_ids, pop_ranks = self.popularity.rank_for_user(user_id)
        if not game_ids:
            return pd.DataFrame(columns=["game_id", "score"])
        content_predictions = self.content_model.predict_many(user_id, game_ids)
        content_ranks = _percentile_ranks(content_predictions)
        blended = (self.content_weight * content_ranks +
                   (1.0 - self.content_weight) * pop_ranks)
        order = np.argsort(-blended, kind="stable")[:n]
        return pd.DataFrame({"game_id": [game_ids[i] for i in order],
                             "score": blended[order]})
