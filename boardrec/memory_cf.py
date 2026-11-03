"""Memory-based Collaborative Filtering (user-based / item-based kNN).

Only TRAIN ratings are used in ``fit``. Hyperparameters (kind, similarity, k)
are chosen by an external Validation loop; Test is never read here.

Prediction (both kinds) starts from the same train-only baseline used by the
SVD model, then adds a similarity-weighted average of neighbour residuals:

    b_ui = mean_u + item_bias_i          (item_bias: shrunken mean deviation)
    user-based: r_ui = b_ui + sum_v s(u,v) (r_vi - b_vi) / sum_v s(u,v)
                over the k most similar users v who rated i in TRAIN
    item-based: r_ui = b_ui + sum_j s(i,j) (r_uj - b_uj) / sum_j s(i,j)
                over the k most similar games j that u rated in TRAIN

Only positive similarities are used. When no neighbour exists the prediction
is the baseline. Item bias is an average rating *deviation*, not a count, so
no popularity score is added. Unknown users fall back to global mean + item
bias (same rule as ``SVDRecommender``).

Similarities are computed over co-rated entries only (missing != 0 stars):
    cosine  : cosine of raw 1–10 ratings on co-rated entries
    pearson : cosine of mean-centred ratings on co-rated entries
              (user-based: centred by user mean; item-based: by item mean)
``shrinkage`` multiplies a similarity by n_common / (n_common + shrinkage) so
that similarities supported by only a few co-ratings are damped, and pairs
with fewer than ``min_support`` co-ratings get similarity 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KINDS = ("user", "item")
SIMILARITIES = ("cosine", "pearson")


def _co_rated_similarity(values: np.ndarray, mask: np.ndarray, *, min_support: int,
                         shrinkage: float) -> np.ndarray:
    """Row-by-row similarity over co-rated columns (dense float32 BLAS).

    ``values`` must be 0 where ``mask`` is 0. Returns a float32 square matrix
    with a zero diagonal.
    """
    x = values.astype(np.float32, copy=False)
    m = mask.astype(np.float32, copy=False)
    numer = x @ x.T
    sq = (x * x) @ m.T                     # sq[a, b] = sum over co-rated of x_a^2
    denom = np.sqrt(sq * sq.T)
    sims = np.zeros_like(numer)
    np.divide(numer, denom, out=sims, where=denom > 1e-12)
    support = m @ m.T
    sims[support < min_support] = 0.0
    if shrinkage > 0:
        sims *= support / (support + np.float32(shrinkage))
    np.fill_diagonal(sims, 0.0)
    return sims


def _weighted_topk(sims: np.ndarray, resid: np.ndarray, valid: np.ndarray, k: int):
    """Per row: weighted mean of ``resid`` over the k largest positive ``sims``
    whose ``valid`` flag is set. Returns (adjustment, has_neighbour)."""
    w = np.where(valid & (sims > 0), sims, 0.0).astype(np.float64)
    n_cols = w.shape[1]
    if k < n_cols:
        drop = np.argpartition(w, n_cols - k, axis=1)[:, :n_cols - k]
        np.put_along_axis(w, drop, 0.0, axis=1)
    denom = w.sum(axis=1)
    numer = (w * resid).sum(axis=1)
    adj = np.zeros(len(denom))
    np.divide(numer, denom, out=adj, where=denom > 1e-12)
    return adj, denom > 1e-12


@dataclass
class MemoryCFRecommender:
    """kNN collaborative filtering for explicit 1–10 ratings.

    Parameters
    ----------
    kind : {'user', 'item'}
    similarity : {'cosine', 'pearson'}
    k : int
        Number of neighbours used for each prediction.
    min_support : int
        Minimum number of co-rated entries for a non-zero similarity.
    shrinkage : float
        Damps similarities with few co-ratings (0 disables).
    item_reg : float
        Shrinkage of the baseline item bias (same meaning as in SVD).

    API: fit(train, games=None), predict, predict_many, recommend.
    """

    kind: str = "item"
    similarity: str = "pearson"
    k: int = 40
    min_support: int = 3
    shrinkage: float = 10.0
    item_reg: float = 10.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.similarity not in SIMILARITIES:
            raise ValueError(f"similarity must be one of {SIMILARITIES}")
        if self.k < 1:
            raise ValueError("k must be positive")
        if self.min_support < 1:
            raise ValueError("min_support must be >= 1")
        if self.shrinkage < 0 or self.item_reg < 0:
            raise ValueError("shrinkage and item_reg must be nonnegative")
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, games: pd.DataFrame | None = None):
        required = {"user_id", "game_id", "rating"}
        if not required.issubset(train.columns):
            raise ValueError(f"train missing columns: {sorted(required - set(train.columns))}")
        if train.empty:
            raise ValueError("Cannot fit on empty ratings")
        frame = train[["user_id", "game_id", "rating"]].copy()
        frame["user_id"] = frame.user_id.astype(str)
        frame["game_id"] = frame.game_id.astype(str)
        frame["rating"] = pd.to_numeric(frame.rating, errors="coerce")
        if frame.rating.isna().any() or not frame.rating.between(1, 10).all():
            raise ValueError("Train ratings must be numeric in the range [1, 10]")
        if frame.duplicated(["user_id", "game_id"]).any():
            raise ValueError("Duplicate (user_id, game_id) pairs; deduplicate first")

        if games is None:
            catalog = set(frame.game_id)
        elif isinstance(games, pd.DataFrame):
            if "game_id" not in games:
                raise ValueError("games metadata must have game_id column")
            catalog = set(games.game_id.astype(str))
        else:
            catalog = {str(g) for g in games}
        if not set(frame.game_id).issubset(catalog):
            raise ValueError("Training games are missing from the supplied catalog")

        self.user_ids_ = pd.Index(sorted(frame.user_id.unique()), name="user_id")
        self.game_ids_ = pd.Index(sorted(catalog), name="game_id")
        self.user_positions_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.game_positions_ = {g: i for i, g in enumerate(self.game_ids_)}
        uidx = self.user_ids_.get_indexer(frame.user_id)
        iidx = self.game_ids_.get_indexer(frame.game_id)
        ratings = frame.rating.to_numpy(dtype=np.float64)
        n_users, n_items = len(self.user_ids_), len(self.game_ids_)

        self.global_mean_ = float(ratings.mean())
        counts = np.bincount(uidx, minlength=n_users)
        self.user_means_ = np.bincount(uidx, weights=ratings, minlength=n_users) / np.maximum(counts, 1)
        dev = ratings - self.user_means_[uidx]
        item_counts = np.bincount(iidx, minlength=n_items)
        self.item_bias_ = np.bincount(iidx, weights=dev, minlength=n_items) / (item_counts + self.item_reg)

        mask = np.zeros((n_users, n_items), dtype=bool)
        mask[uidx, iidx] = True
        rmat = np.zeros((n_users, n_items), dtype=np.float32)
        rmat[uidx, iidx] = ratings
        resid = np.zeros((n_users, n_items), dtype=np.float32)
        resid[uidx, iidx] = ratings - self.user_means_[uidx] - self.item_bias_[iidx]
        self.mask_ = mask
        self.residuals_ = resid

        if self.similarity == "cosine":
            sim_values = rmat
        elif self.kind == "user":
            sim_values = np.zeros_like(rmat)
            sim_values[uidx, iidx] = ratings - self.user_means_[uidx]
        else:
            item_means = np.bincount(iidx, weights=ratings, minlength=n_items) / np.maximum(item_counts, 1)
            sim_values = np.zeros_like(rmat)
            sim_values[uidx, iidx] = ratings - item_means[iidx]

        if self.kind == "user":
            self.similarity_ = _co_rated_similarity(sim_values, mask, min_support=self.min_support,
                                                    shrinkage=self.shrinkage)
        else:
            self.similarity_ = _co_rated_similarity(sim_values.T, mask.T, min_support=self.min_support,
                                                    shrinkage=self.shrinkage)
        self.fitted_ = True
        return self

    # -------------------------------------------------------------- predict
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
        items = np.array([self.game_positions_[g] for g in requested])
        u = self.user_positions_.get(str(user_id))
        if u is None:
            return np.clip(self.global_mean_ + self.item_bias_[items], 1.0, 10.0)
        preds = self.user_means_[u] + self.item_bias_[items]
        if self.kind == "user":
            # rows = candidate items, columns = all neighbour users
            sims = np.broadcast_to(self.similarity_[u], (len(items), self.similarity_.shape[0]))
            adj, _ = _weighted_topk(sims, self.residuals_[:, items].T, self.mask_[:, items].T, self.k)
        else:
            rated = np.flatnonzero(self.mask_[u])
            if rated.size == 0:
                return np.clip(preds, 1.0, 10.0)
            sims = self.similarity_[np.ix_(items, rated)]
            resid = np.broadcast_to(self.residuals_[u, rated], sims.shape)
            adj, _ = _weighted_topk(sims, resid, np.ones_like(sims, dtype=bool), self.k)
        return np.clip(preds + adj, 1.0, 10.0)

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit before recommend")
        if n < 1:
            raise ValueError("n must be positive")
        u = self.user_positions_.get(str(user_id))
        if u is None:
            candidates = list(self.game_ids_)
        else:
            candidates = [g for g, seen in zip(self.game_ids_, self.mask_[u]) if not seen]
        scores = self.predict_many(user_id, candidates)
        result = pd.DataFrame({"game_id": candidates, "score": scores})
        return (result.sort_values(["score", "game_id"], ascending=[False, True], kind="stable")
                .head(n).reset_index(drop=True))
