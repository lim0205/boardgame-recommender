"""Reproducible, chunked preparation of rating observations.

Standardized outputs have columns: user_id (str), game_id (str), rating (float).
Note: ranking game popularity prior to splitting is a dataset-selection policy,
not a statistic used for model training. The same fixed subset is shared by
all subsequent recommenders. Never derive a user profile from held-out ratings.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

RATING_ALIASES = {
    "user_id": ["user_id", "userid", "username", "user", "member"],
    "game_id": ["game_id", "gameid", "bggid", "bgg_id", "item_id"],
    "rating": ["rating", "ratingvalue", "user_rating", "score"],
}
GAME_ALIASES = ["game_id", "gameid", "bggid", "bgg_id", "item_id"]


@dataclass(frozen=True)
class PrepConfig:
    top_games: int = 1000
    min_user_ratings: int = 20
    max_users: int | None = 20000
    seed: int = 42
    chunksize: int = 200000
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    def __post_init__(self):
        if self.top_games < 1 or self.min_user_ratings < 3:
            raise ValueError("top_games >=1 and min_user_ratings >=3 required")
        if self.max_users is not None and self.max_users < 1:
            raise ValueError("max_users must be positive")
        if self.chunksize < 1:
            raise ValueError("chunksize must be positive")
        if not (0 < self.train_ratio < 1 and 0 < self.val_ratio < 1 and
                self.train_ratio + self.val_ratio < 1):
            raise ValueError("train/val ratios must be positive and sum below one")


def _name_map(columns: list[str], aliases: dict[str, list[str]]) -> dict[str, str]:
    norm = {str(c).strip().lower().replace(" ", "").replace("-", ""): c for c in columns}
    out = {}
    for field, options in aliases.items():
        for name in options:
            if name in norm:
                out[norm[name]] = field
                break
        else:
            raise ValueError(f"Missing {field}; available columns: {columns}")
    return out


def _clean_ratings(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns=_name_map(list(chunk.columns), RATING_ALIASES))
    x = chunk[["user_id", "game_id", "rating"]].copy()
    x = x.dropna(subset=["user_id", "game_id", "rating"])
    x["user_id"] = x.user_id.astype("string").str.strip()
    x["game_id"] = x.game_id.astype("string").str.strip()
    x["rating"] = pd.to_numeric(x.rating, errors="coerce")
    x = x[x.rating.between(1.0, 10.0) & x.user_id.ne("") & x.game_id.ne("")].copy()
    return x.astype({"user_id": "str", "game_id": "str", "rating": "float64"})


def read_ratings_chunks(path: str | Path, chunksize: int):
    # Retain string identifiers, even if user names are numeric-like.
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, dtype=str):
        yield _clean_ratings(chunk)


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ID as game_id, normalize IDs, and remove duplicate game records."""
    found = _name_map(list(df.columns), {"game_id": GAME_ALIASES})
    out = df.rename(columns=found).copy()
    out["game_id"] = out.game_id.astype("string").str.strip().astype(str)
    out = out[out.game_id.ne("") & out.game_id.ne("<NA>")]
    return out.drop_duplicates("game_id").reset_index(drop=True)


def split_by_user(ratings: pd.DataFrame, cfg: PrepConfig) -> dict[str, pd.DataFrame]:
    """Deterministic per-user 80/10/10 (approximately) with ≥1 held-out row.

    Return independent frames with NO split column; all partitions have the
    canonical columns (user_id, game_id, rating).
    """
    x = ratings.sort_values(["user_id", "game_id"], kind="stable").reset_index(drop=True)
    assignment = np.full(len(x), "train", dtype="<U5")
    rng = np.random.default_rng(cfg.seed)
    for _, idx in x.groupby("user_id", sort=True).indices.items():
        n = len(idx)
        if n < 3:
            raise ValueError("Each user must have at least 3 observations")
        n_val = max(1, int(round(n * cfg.val_ratio)))
        n_test = max(1, int(round(n * (1 - cfg.train_ratio - cfg.val_ratio))))
        if n_val + n_test >= n:
            n_val, n_test = 1, 1
        order = rng.permutation(idx)
        assignment[order[:n_val]] = "val"
        assignment[order[n_val:n_val + n_test]] = "test"
    return {k: x.loc[assignment == k, ["user_id", "game_id", "rating"]].reset_index(drop=True)
            for k in ("train", "val", "test")}


def prepare(ratings_path: str | Path, games_path: str | Path,
            mechanics_path: str | Path, themes_path: str | Path,
            output_dir: str | Path, cfg: PrepConfig = PrepConfig()) -> dict:
    """Read large CSV in passes; keep only selected users/games in memory.

    All four source files are required. Does not download external data.
    """
    inputs = [Path(p) for p in (ratings_path, games_path, mechanics_path, themes_path)]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Pass 1: pick a small game catalog by number of valid observations.
    game_counts: dict[str, int] = {}
    for part in read_ratings_chunks(inputs[0], cfg.chunksize):
        for game, count in part.game_id.value_counts().items():
            game_counts[game] = game_counts.get(game, 0) + int(count)
    if not game_counts:
        raise ValueError("No valid ratings in file")
    top_games = sorted(game_counts, key=lambda g: (-game_counts[g], g))[:cfg.top_games]

    # Restrict catalog to items with metadata. Avoid re-loading full ratings.
    games = normalize_metadata(pd.read_csv(inputs[1], low_memory=False))
    mechanics = normalize_metadata(pd.read_csv(inputs[2], low_memory=False))
    themes = normalize_metadata(pd.read_csv(inputs[3], low_memory=False))
    available = set(games.game_id) & set(mechanics.game_id) & set(themes.game_id)
    selected_games = set(top_games) & available
    if not selected_games:
        raise ValueError("No common game IDs across four CSV files")
    LOGGER.info("Selected %d common games", len(selected_games))

    # Pass 2: count ratings by user after choosing the game catalog.
    user_counts: dict[str, int] = {}
    for part in read_ratings_chunks(inputs[0], cfg.chunksize):
        part = part.loc[part.game_id.isin(selected_games)]
        for user, count in part.user_id.value_counts().items():
            user_counts[user] = user_counts.get(user, 0) + int(count)
    eligible = sorted(u for u, count in user_counts.items() if count >= cfg.min_user_ratings)
    if not eligible:
        raise ValueError("No eligible users: lower --min-user-ratings or --top-games")
    if cfg.max_users is not None and len(eligible) > cfg.max_users:
        rng = np.random.default_rng(cfg.seed)
        eligible = sorted(rng.choice(eligible, cfg.max_users, replace=False).tolist())
    selected_users = set(eligible)

    # Pass 3: retain only the chosen user × game records.
    parts = []
    for part in read_ratings_chunks(inputs[0], cfg.chunksize):
        part = part.loc[part.game_id.isin(selected_games) & part.user_id.isin(selected_users)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError("No ratings remained after filtering")
    chosen = pd.concat(parts, ignore_index=True)
    # Resolve duplicate (user, game) records deterministically via mean.
    chosen = chosen.groupby(["user_id", "game_id"], as_index=False, sort=True).rating.mean()
    chosen_users = chosen.user_id.value_counts()
    chosen = chosen[chosen.user_id.isin(chosen_users[chosen_users >= cfg.min_user_ratings].index)]
    if chosen.empty:
        raise ValueError("After deduplicating, no user meets the minimum rating count")
    splits = split_by_user(chosen, cfg)
    for split_name, frame in splits.items():
        frame.to_csv(output / f"{split_name}.csv", index=False)
    kept_ids = set(chosen.game_id)
    for filename, frame in (("games", games), ("mechanics", mechanics), ("themes", themes)):
        frame.loc[frame.game_id.isin(kept_ids)].sort_values("game_id").to_csv(
            output / f"{filename}.csv", index=False)
    manifest = {"config": asdict(cfg), "source": [p.name for p in inputs],
                "n_users": int(chosen.user_id.nunique()), "n_games": int(chosen.game_id.nunique()),
                "n_ratings": len(chosen), "split_counts": {k: len(v) for k, v in splits.items()},
                "data_note": "Subset/split counts are for provided files, not global BGG stats."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_prepared(data_dir: str | Path):
    p = Path(data_dir)
    return tuple(pd.read_csv(p / name, dtype={"user_id": str, "game_id": str} if name in
                                  ("train.csv", "val.csv", "test.csv") else {"game_id": str})
                 for name in ("train.csv", "val.csv", "test.csv", "games.csv",
                              "mechanics.csv", "themes.csv"))
