"""Adapter to make a teammate's rating predictor usable by the shared evaluator.

The source recommender only needs a function predict(user_id, game_id) -> float.
The adapter provides the group's common predict_many and recommend methods.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictAdapter:
    """Wrap any callable `(user_id, game_id) -> predicted rating`.

    Parameters
    ----------
    predict_fn: callable
        Method from teammate's fitted model. Must return a 1-10 rating.
    train: pd.DataFrame
        Canonical training rows. Used ONLY to exclude already-rated games.
    catalog_game_ids: iterable
        Same game IDs that the content model considers for recommending.
    """

    predict_fn: Callable[[str, str], float]
    train: pd.DataFrame
    catalog_game_ids: Iterable[str]

    def __post_init__(self):
        required = {"user_id", "game_id", "rating"}
        if not required.issubset(self.train.columns):
            raise ValueError(f"train must contain {required}")
        self.catalog_ = sorted({str(g) for g in self.catalog_game_ids})
        if not self.catalog_:
            raise ValueError("catalog_game_ids cannot be empty")
        train = self.train.assign(user_id=self.train.user_id.astype(str),
                                  game_id=self.train.game_id.astype(str))
        self.rated_ = {str(uid): set(group.game_id) for uid, group in train.groupby("user_id")}

    def predict(self, user_id: str, game_id: str) -> float:
        result = float(self.predict_fn(str(user_id), str(game_id)))
        if not np.isfinite(result):
            raise ValueError("predict_fn returned a non-finite value")
        return float(np.clip(result, 1.0, 10.0))

    def predict_many(self, user_id: str, game_ids) -> np.ndarray:
        return np.asarray([self.predict(user_id, str(game)) for game in game_ids], dtype=float)

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n must be positive")
        rated = self.rated_.get(str(user_id), set())
        candidates = [g for g in self.catalog_ if g not in rated]
        predictions = self.predict_many(str(user_id), candidates)
        return (pd.DataFrame({"game_id": candidates, "score": predictions})
                .sort_values(["score", "game_id"], ascending=[False, True], kind="stable")
                .head(n).reset_index(drop=True))
