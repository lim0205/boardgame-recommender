"""Model-based Collaborative Filtering with *truncated SVD*.

Fit a sparse matrix of observed rating residuals (unobserved entries are zero
residual, NOT zero-star ratings), and reconstruct a low-rank approximation.
This is classical truncated SVD; it is not Surprise's SGD/Funk SVD.

Prediction: mean_train_rating_of_user + shrunken_item_bias +
            residual_scale * (U_k Sigma_k V_k.T)[user, game].

Only TRAIN ratings are used in fit. The fitting code never reads validation
or test ratings; an external validation loop chooses hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


@dataclass
class SVDRecommender:
    """SVD matrix-factorization recommender for explicit 1–10 ratings.

    Parameters
    ----------
    n_factors : int
        Number of singular components (effective k is bounded by matrix size).
    item_reg : float
        Shrinkage applied to per-item rating-offset estimates.
    residual_scale : float
        Strength of the truncated-SVD residual reconstruction.
    seed : int
        Random seed for the sparse SVD initialization.

    API matches the project's Content-based model and PredictAdapter:
        fit(train, games=None) -> self
        predict(user_id, game_id) -> float
        predict_many(user_id, game_ids) -> numpy.ndarray
        recommend(user_id, n=10) -> DataFrame[game_id, score]
    """

    n_factors: int = 20
    item_reg: float = 10.0
    residual_scale: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.n_factors < 1:
            raise ValueError("n_factors must be positive")
        if self.item_reg < 0:
            raise ValueError("item_reg must be nonnegative")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be nonnegative")
        self.fitted_ = False

    def fit(self, train: pd.DataFrame, games: pd.DataFrame | None = None):
        required = {"user_id", "game_id", "rating"}
        if not required.issubset(train.columns):
            raise ValueError(f"train missing columns: {sorted(required - set(train.columns))}")
        if train.empty:
            raise ValueError("Cannot fit on empty ratings")
        frame = train[["user_id", "game_id", "rating"]].copy()
        frame.user_id = frame.user_id.astype(str)
        frame.game_id = frame.game_id.astype(str)
        frame.rating = pd.to_numeric(frame.rating, errors="coerce")
        if frame.rating.isna().any() or not frame.rating.between(1, 10).all():
            raise ValueError("Train ratings must be numeric in the range [1, 10]")
        if frame.duplicated(["user_id", "game_id"]).any():
            raise ValueError("Duplicate (user_id, game_id) pairs; deduplicate first")

        self.user_ids_ = pd.Index(sorted(frame.user_id.unique()), name="user_id")
        if games is None:
            item_catalog = set(frame.game_id)
        elif isinstance(games, pd.DataFrame):
            if "game_id" not in games:
                raise ValueError("games metadata must have game_id column")
            item_catalog = set(games.game_id.astype(str))
        else:
            item_catalog = set(str(x) for x in games)
        if not set(frame.game_id).issubset(item_catalog):
            raise ValueError("Training games are missing from the supplied catalog")
        self.game_ids_ = pd.Index(sorted(item_catalog), name="game_id")
        self.user_positions_ = {u: idx for idx, u in enumerate(self.user_ids_)}
        self.game_positions_ = {g: idx for idx, g in enumerate(self.game_ids_)}
        uidx = self.user_ids_.get_indexer(frame.user_id)
        iidx = self.game_ids_.get_indexer(frame.game_id)
        ratings = frame.rating.to_numpy(dtype=np.float64)
        n_users, n_items = len(self.user_ids_), len(self.game_ids_)
        self.global_mean_ = float(np.mean(ratings))

        # Zero ratings are never used for unknown interactions. Missing data
        # is represented as zero *residual* after centering.
        user_counts = np.bincount(uidx, minlength=n_users)
        means = np.bincount(uidx, weights=ratings, minlength=n_users) / np.maximum(user_counts, 1)
        self.user_means_ = means
        self.seen_ = {}
        for user, group in frame.groupby("user_id", sort=False):
            self.seen_[str(user)] = set(group.game_id.astype(str))

        deviations = ratings - means[uidx]
        item_counts = np.bincount(iidx, minlength=n_items)
        bias_sum = np.bincount(iidx, weights=deviations, minlength=n_items)
        self.item_bias_ = bias_sum / (item_counts + self.item_reg)
        residual = deviations - self.item_bias_[iidx]
        sparse_residuals = csr_matrix((residual, (uidx, iidx)),
                                      shape=(n_users, n_items), dtype=np.float64)
        available = min(n_users, n_items) - 1
        self.effective_factors_ = min(self.n_factors, max(available, 0))
        if self.effective_factors_:
            u, singular, vt = svds(sparse_residuals, k=self.effective_factors_,
                                   which="LM", random_state=self.seed)
            # For reconstruction, use symmetric sqrt(sigma) factorization.
            order = np.argsort(singular)[::-1]
            sqrt_singular = np.sqrt(np.maximum(singular[order], 0.0))
            self.user_factors_ = u[:, order] * sqrt_singular
            self.item_factors_ = vt[order, :].T * sqrt_singular
        else:
            self.user_factors_ = np.zeros((n_users, 0))
            self.item_factors_ = np.zeros((n_items, 0))
        self.fitted_ = True
        return self

    def predict(self, user_id: str, game_id: str) -> float:
        return float(self.predict_many(user_id, [game_id])[0])

    def predict_many(self, user_id: str, game_ids) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Call fit before predict")
        requested = [str(g) for g in game_ids]
        if not requested:
            return np.array([], dtype=float)
        missing = [g for g in requested if g not in self.game_positions_]
        if missing:
            raise KeyError(f"Unknown game(s): {missing[:5]}")
        indices = np.array([self.game_positions_[g] for g in requested])
        u = self.user_positions_.get(str(user_id))
        mean = self.global_mean_ if u is None else self.user_means_[u]
        preds = mean + self.item_bias_[indices]
        if u is not None and self.effective_factors_:
            preds = preds + self.residual_scale * (self.item_factors_[indices] @ self.user_factors_[u])
        return np.clip(preds, 1.0, 10.0)

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit before recommend")
        if n < 1:
            raise ValueError("n must be positive")
        seen = self.seen_.get(str(user_id), set())
        candidates = [g for g in self.game_ids_ if g not in seen]
        scores = self.predict_many(user_id, candidates)
        result = pd.DataFrame({"game_id": candidates, "score": scores})
        return (result.sort_values(["score", "game_id"], ascending=[False, True], kind="stable")
                .head(n).reset_index(drop=True))
