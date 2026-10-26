"""Importer for the *actual file layout* of the Kaggle threnjen BGG dataset.

This module does not download data or authorize its use. Dataset user must
confirm rights before applying the recommender to data from BGG.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, BadZipFile
import shutil
import pandas as pd

EXPECTED = ("games.csv", "user_ratings.csv", "mechanics.csv", "themes.csv")


def find_csvs(directory: str | Path) -> dict[str, Path]:
    """Find required CSVs even when they are under a Kaggle extraction folder."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"CSV folder not found: {root}")
    indexed: dict[str, list[Path]] = {name: [] for name in EXPECTED}
    for p in root.rglob("*.csv"):
        key = p.name.lower()
        if key in indexed:
            indexed[key].append(p)
    missing = [name for name, files in indexed.items() if not files]
    duplicate = [name for name, files in indexed.items() if len(files) > 1]
    if missing or duplicate:
        raise ValueError(f"BGG CSV discovery failed. missing={missing}; duplicate={duplicate}. "
                         "Expected games.csv, user_ratings.csv, mechanics.csv, themes.csv")
    return {k: v[0] for k, v in indexed.items()}


def extract_required_csvs(archive: str | Path, destination: str | Path) -> dict[str, Path]:
    """Stream just the four needed CSVs from a Kaggle ZIP; reject ambiguity.

    Write by known basename (never trust ZIP internal path); avoid reading
    the large ratings CSV entirely into RAM.
    """
    archive, destination = Path(archive), Path(destination)
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {archive}")
    try:
        with ZipFile(archive) as zf:
            found = {name: [] for name in EXPECTED}
            for item in zf.infolist():
                if not item.is_dir() and Path(item.filename).name.lower() in found:
                    found[Path(item.filename).name.lower()].append(item)
            missing = [k for k, items in found.items() if not items]
            duplicate = [k for k, items in found.items() if len(items) > 1]
            if missing or duplicate:
                raise ValueError(f"Kaggle ZIP format mismatch: missing={missing}, duplicate={duplicate}")
            destination.mkdir(parents=True, exist_ok=True)
            extracted = {}
            for name in EXPECTED:
                output = destination / name
                with zf.open(found[name][0]) as src, output.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                extracted[name] = output
    except BadZipFile as exc:
        raise ValueError(f"Not a readable dataset ZIP: {archive}") from exc
    return extracted


def check_bgg_schema(paths: dict[str, Path]) -> dict[str, list[str]]:
    """Validate original dataset column identities before any model is run."""
    headings = {}
    for name in EXPECTED:
        path = paths[name]
        headings[name] = [str(col) for col in pd.read_csv(path, nrows=0).columns]
        lower = {s.strip().lower() for s in headings[name]}
        if "bggid" not in lower and "bgg_id" not in lower and "game_id" not in lower:
            raise ValueError(f"{name}: no BGGId or game ID in {headings[name]}")
        if name == "user_ratings.csv":
            if not lower.intersection({"username", "user_id", "userid"}):
                raise ValueError("user_ratings.csv: missing Username/user_id")
            if not lower.intersection({"rating", "score"}):
                raise ValueError("user_ratings.csv: missing Rating")
    return headings
