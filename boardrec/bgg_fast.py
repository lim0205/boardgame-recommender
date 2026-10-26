"""Efficient preprocessing for the BGG CSV layout (one typed read).

For low-memory environments, see preprocessing.prepare (chunked three-pass).
Only derived/split CSVs are written locally; they must never be committed.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import gc
import json
import logging

import numpy as np
import pandas as pd
from .preprocessing import PrepConfig, normalize_metadata, split_by_user

LOG = logging.getLogger(__name__)


def prepare_bgg_fast(ratings_path: str | Path, games_path: str | Path,
                     mechanics_path: str | Path, themes_path: str | Path,
                     output_dir: str | Path, cfg: PrepConfig = PrepConfig()) -> dict:
    """Prepare BGG explicit ratings; metadata joins use the BGGId field.

    Designed for machines with >=4 GiB of free RAM. Game selection is based
    on global rating counts; see README for offline evaluation limitations.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ratings_path,
                       usecols=["BGGId", "Username", "Rating"],
                       dtype={"BGGId": "Int32", "Username": "string", "Rating": "float32"},
                       low_memory=False)
    n_raw = len(data)
    LOG.info("Loaded %d raw rating records", n_raw)
    valid = data.BGGId.notna() & data.Username.notna() & data.Rating.between(1, 10)
    data = data.loc[valid].copy()
    data.Username = data.Username.str.strip()
    data = data.loc[data.Username.ne("")].copy()
    n_valid = len(data)
    LOG.info("%d ratings passed data validation", n_valid)
    top_games = set(data.BGGId.value_counts().head(cfg.top_games).index.astype(int).tolist())

    games = normalize_metadata(pd.read_csv(games_path, low_memory=False))
    mechanics = normalize_metadata(pd.read_csv(mechanics_path, low_memory=False))
    themes = normalize_metadata(pd.read_csv(themes_path, low_memory=False))
    available = set(games.game_id) & set(mechanics.game_id) & set(themes.game_id)
    selected_games = top_games & {int(i) for i in available if str(i).isdigit()}
    data = data.loc[data.BGGId.isin(selected_games)].copy()
    LOG.info("%d records for %d chosen games", len(data), len(selected_games))
    counts = data.Username.value_counts()
    eligible = sorted(counts.loc[counts >= cfg.min_user_ratings].index.tolist())
    eligible_before_sample = len(eligible)
    if not eligible:
        raise ValueError("No eligible users; lower min_user_ratings or increase top_games")
    if cfg.max_users is not None and len(eligible) > cfg.max_users:
        eligible = sorted(np.random.default_rng(cfg.seed).choice(
            eligible, cfg.max_users, replace=False).tolist())
    data = data.loc[data.Username.isin(set(eligible))].copy()
    LOG.info("%d records after sampling %d users", len(data), len(eligible))

    chosen = pd.DataFrame({
        "user_id": data.Username.astype(str).to_numpy(),
        "game_id": data.BGGId.astype("str").to_numpy(),
        "rating": data.Rating.astype(float).to_numpy(),
    })
    del data
    gc.collect()
    chosen = chosen.groupby(["user_id", "game_id"], as_index=False, sort=True).rating.mean()
    counts = chosen.user_id.value_counts()
    chosen = chosen.loc[chosen.user_id.isin(
        counts.loc[counts >= cfg.min_user_ratings].index)].copy()
    if chosen.empty:
        raise ValueError("No eligible users after deduplicating ratings")

    splits = split_by_user(chosen, cfg)
    for name, subset in splits.items():
        subset.to_csv(output / f"{name}.csv", index=False)
    ids = set(chosen.game_id)
    for name, frame in (("games", games), ("mechanics", mechanics), ("themes", themes)):
        frame.loc[frame.game_id.isin(ids)].sort_values("game_id").to_csv(
            output / f"{name}.csv", index=False)
    manifest = {
        "config": asdict(cfg),
        "source": [Path(f).name for f in (ratings_path, games_path, mechanics_path, themes_path)],
        "n_raw_ratings": n_raw,
        "n_valid_ratings": n_valid,
        "eligible_users_before_sampling": eligible_before_sample,
        "n_users": int(chosen.user_id.nunique()),
        "n_games": int(chosen.game_id.nunique()),
        "n_ratings": len(chosen),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "data_note": "BGG-format data supplied by the user; no raw ratings uploaded to GitHub.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
