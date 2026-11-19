"""Sparse content-based recommender with a teammate-compatible prediction API.

No N×N similarity matrix is created. Cosine similarities are computed on demand.
Rating prediction uses the user's rating mean plus an item-similarity-weighted
average of their deviations from that mean. Unmatched candidates fall back to
user mean; new users fall back to global train mean (not personalized).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from .preprocessing import normalize_metadata

FEATURE_SETS = ("mechanics", "themes", "both")


def _binary_features(metadata: pd.DataFrame, game_ids: pd.Index, prefix: str):
    df = normalize_metadata(metadata).set_index("game_id").reindex(game_ids)
    columns = [col for col in df.columns if col != "game_id"]
    # One-hot tables; strictly exclude string and ID columns and drop all-zero
    # columns. `0/1` is a *feature*, not a zero rating.
    values = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = (values > 0).astype(np.float32)
    values = values.loc[:, values.any(axis=0)]
    if values.shape[1] == 0:
        return sparse.csr_matrix((len(game_ids), 0), dtype=np.float32), []
    return sparse.csr_matrix(values.to_numpy()), [f"{prefix}{c}" for c in values.columns]


@dataclass
class ContentBasedRecommender:
    """Cosine-similarity recommender for explicit 1–10 ratings.

    Parameters
    ----------
    feature_set : {'mechanics', 'themes', 'both'}
        Features derived from game metadata, independent of user ratings.

    Public interface
    ----------------
    fit(train, games, mechanics, themes) -> self
    predict(user_id, game_id) -> float
    predict_many(user_id, game_ids) -> np.ndarray
    recommend(user_id, n=10) -> pd.DataFrame[game_id, score]
    """
    feature_set: str = "both"

    def __post_init__(self):
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.fitted_ = False

    def fit(self, train: pd.DataFrame, games: pd.DataFrame,
            mechanics: pd.DataFrame, themes: pd.DataFrame):
        required = {"user_id", "game_id", "rating"}
        if not required.issubset(train.columns):
            raise ValueError(f"train must contain {required}")
        train = train.copy()
        train["user_id"] = train.user_id.astype(str)
        train["game_id"] = train.game_id.astype(str)
        train["rating"] = pd.to_numeric(train.rating, errors="coerce")
        if train.empty or train.rating.isna().any() or not train.rating.between(1, 10).all():
            raise ValueError("train ratings must be nonempty, numeric and between 1 and 10")
        if train.duplicated(["user_id", "game_id"]).any():
            raise ValueError("duplicate (user_id, game_id) ratings; deduplicate first")

        catalog = normalize_metadata(games)
        allowed = set(normalize_metadata(mechanics).game_id) & set(normalize_metadata(themes).game_id)
        allowed &= set(catalog.game_id)
        game_ids = sorted(allowed)
        if not game_ids:
            raise ValueError("No common games in games/mechanics/themes metadata")
        self.game_ids_ = pd.Index(game_ids, name="game_id")
        self.game_positions_ = {g: i for i, g in enumerate(game_ids)}
        self.games_ = catalog.set_index("game_id").reindex(self.game_ids_)

        m_mat, m_cols = _binary_features(mechanics, self.game_ids_, "mech__")
        t_mat, t_cols = _binary_features(themes, self.game_ids_, "theme__")
        if self.feature_set == "mechanics":
            matrix, columns = m_mat, m_cols
        elif self.feature_set == "themes":
            matrix, columns = t_mat, t_cols
        else:
            matrix = sparse.hstack([m_mat, t_mat], format="csr")
            columns = m_cols + t_cols
        if matrix.shape[1] == 0:
            raise ValueError("No nonzero binary features in selected metadata")
        # sklearn.normalize: unit L2 length per game (axis=1) makes dot products cosine scores.
        self.features_ = normalize(matrix, norm="l2", axis=1).tocsr()
        self.feature_names_ = columns
        self.global_mean_ = float(train.rating.mean())
        # Only TRAIN ratings build each user's rated-item profile and mean.
        self.user_profiles_: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for user, df in train.groupby("user_id", sort=False):
            df = df[df.game_id.isin(self.game_positions_)]
            if df.empty:
                continue
            positions = np.array([self.game_positions_[g] for g in df.game_id], dtype=int)
            ratings = df.rating.to_numpy(dtype=float)
            self.user_profiles_[str(user)] = (positions, ratings, float(ratings.mean()))
        self.fitted_ = True
        return self

    def predict(self, user_id: str, game_id: str) -> float:
        return float(self.predict_many(user_id, [game_id])[0])

    def predict_many(self, user_id: str, game_ids) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Call fit() before predict()")
        requested = [str(g) for g in game_ids]
        if any(g not in self.game_positions_ for g in requested):
            unknown = [g for g in requested if g not in self.game_positions_]
            raise KeyError(f"Unknown game_id(s): {unknown[:5]}")
        if not requested:
            return np.array([], dtype=float)
        profile = self.user_profiles_.get(str(user_id))
        if profile is None:
            return np.full(len(requested), self.global_mean_, dtype=float)
        rated_pos, rated_scores, user_mean = profile
        requested_pos = [self.game_positions_[g] for g in requested]
        # Compute candidate-to-history similarity on demand; avoid dense game x game storage.
        sims = (self.features_[requested_pos] @ self.features_[rated_pos].T).toarray()
        sims = np.maximum(sims, 0)
        # Relative preference: a 9 compared with user's mean of 6 is positive.
        numer = sims @ (rated_scores - user_mean)
        denom = sims.sum(axis=1)
        preds = np.full(len(requested), user_mean, dtype=float)
        # np.divide(where=...) avoids zero division; no similar games => user-mean fallback.
        np.divide(numer, denom, out=preds, where=denom > 1e-12)
        # Result in preds currently relative deviation for nonzero denom.
        preds[denom > 1e-12] += user_mean
        return np.clip(preds, 1.0, 10.0)

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        """Recommend unrated (in TRAIN) catalog games, sorted score then ID.

        Note: held-out positives remain candidates, as required for evaluation.
        """
        if n < 1:
            raise ValueError("n must be positive")
        rated = set()
        profile = self.user_profiles_.get(str(user_id))
        if profile is not None:
            rated = {self.game_ids_[p] for p in profile[0]}
        candidates = [g for g in self.game_ids_ if g not in rated]
        scores = self.predict_many(user_id, candidates)
        ranked = pd.DataFrame({"game_id": candidates, "score": scores})
        return ranked.sort_values(["score", "game_id"], ascending=[False, True],
                                  kind="stable").head(n).reset_index(drop=True)
