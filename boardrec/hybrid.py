"""Weighted Hybrid of a Content-based model and a Collaborative Filtering model.

    hybrid_score(u, i) = alpha * content(u, i) + (1 - alpha) * cf(u, i)

Both components predict on the same 1–10 scale and are fitted on TRAIN only.
``alpha`` and the CF component are selected on Validation by an external
loop (see ``hybrid_experiments``). No popularity term is added.

alpha = 1.0 is pure Content-based, alpha = 0.0 is pure CF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HybridRecommender:
    """Linear blend of two already-fitted models with the shared API.

    Parameters
    ----------
    content_model, cf_model : fitted models exposing ``predict_many``.
    alpha : float in [0, 1]
        Weight of the content model.
    train : DataFrame, optional
        TRAIN ratings, used only to exclude already-rated games from
        ``recommend``. If omitted, the content model's train profile is used.
    """

    content_model: object
    cf_model: object
    alpha: float = 0.5
    train: pd.DataFrame | None = None

    def __post_init__(self):
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        content_ids = getattr(self.content_model, "game_ids_", None)
        cf_ids = getattr(self.cf_model, "game_ids_", None)
        if content_ids is None or cf_ids is None:
            raise ValueError("Both component models must be fitted (game_ids_ missing)")
        # Shared candidate catalog so Coverage@10 stays comparable.
        self.game_ids_ = pd.Index(sorted(set(map(str, content_ids)) & set(map(str, cf_ids))),
                                  name="game_id")
        if len(self.game_ids_) == 0:
            raise ValueError("Component models share no games")
        self.rated_: dict[str, set[str]] = {}
        if self.train is not None:
            frame = self.train.assign(user_id=self.train.user_id.astype(str),
                                      game_id=self.train.game_id.astype(str))
            self.rated_ = {u: set(g.game_id) for u, g in frame.groupby("user_id")}
        else:
            profiles = getattr(self.content_model, "user_profiles_", None)
            if profiles is None:
                # Without TRAIN we cannot exclude already-rated games; fail loudly.
                raise ValueError("Pass train= unless content_model is a fitted ContentBasedRecommender")
            ids = self.content_model.game_ids_
            self.rated_ = {u: {ids[p] for p in prof[0]} for u, prof in profiles.items()}

    def predict(self, user_id: str, game_id: str) -> float:
        return float(self.predict_many(user_id, [game_id])[0])

    def predict_many(self, user_id: str, game_ids) -> np.ndarray:
        requested = [str(g) for g in game_ids]
        if not requested:
            return np.array([], dtype=float)
        a = float(self.alpha)
        content = np.asarray(self.content_model.predict_many(str(user_id), requested), dtype=float)
        cf = np.asarray(self.cf_model.predict_many(str(user_id), requested), dtype=float)
        return np.clip(a * content + (1.0 - a) * cf, 1.0, 10.0)

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n must be positive")
        rated = self.rated_.get(str(user_id), set())
        candidates = [g for g in self.game_ids_ if g not in rated]
        scores = self.predict_many(user_id, candidates)
        return (pd.DataFrame({"game_id": candidates, "score": scores})
                .sort_values(["score", "game_id"], ascending=[False, True], kind="stable")
                .head(n).reset_index(drop=True))


class CachedPredictions:
    """Memoise a fitted model's predictions per (user, game).

    Used only by the Validation grid search so that each component model is
    queried once per user/game and re-used for every alpha. Returned values
    are identical to calling the wrapped model directly.
    """

    def __init__(self, model):
        self.model = model
        self.game_ids_ = model.game_ids_
        self._cache: dict[str, dict[str, float]] = {}

    def predict_many(self, user_id: str, game_ids) -> np.ndarray:
        user_id = str(user_id)
        requested = [str(g) for g in game_ids]
        cache = self._cache.setdefault(user_id, {})
        missing = [g for g in dict.fromkeys(requested) if g not in cache]
        if missing:
            values = np.asarray(self.model.predict_many(user_id, missing), dtype=float)
            cache.update(zip(missing, values.tolist()))
        return np.array([cache[g] for g in requested], dtype=float)

    def predict(self, user_id: str, game_id: str) -> float:
        return float(self.predict_many(user_id, [game_id])[0])
