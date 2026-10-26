"""Small, entirely synthetic fixture. NOT derived from BoardGameGeek data."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_demo(path: str | Path, seed: int = 123):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_games, n_users = 60, 45
    game_ids = [str(1000 + i) for i in range(n_games)]
    mechanics = pd.DataFrame({"BGGId": game_ids})
    themes = pd.DataFrame({"BGGId": game_ids})
    for i in range(7):
        mechanics[f"Mechanic_{i}"] = rng.integers(0, 2, n_games)
    for i in range(6):
        themes[f"Theme_{i}"] = rng.integers(0, 2, n_games)
    # Each user rates 25+ games with preference linked to metadata.
    raw = []
    mech = mechanics.iloc[:, 1:].to_numpy(float)
    theme = themes.iloc[:, 1:].to_numpy(float)
    features = np.hstack([mech, theme])
    for u in range(n_users):
        pref = rng.normal(0, 0.9, features.shape[1])
        ratings = 6 + (features @ pref) / 3 + rng.normal(0, 0.8, n_games)
        chosen = rng.choice(n_games, 30, replace=False)
        for i in chosen:
            raw.append((f"demo_user_{u:03d}", game_ids[i], round(float(np.clip(ratings[i], 1, 10)), 1)))
    pd.DataFrame(raw, columns=["Username", "BGGId", "Rating"]).to_csv(path / "user_ratings.csv", index=False)
    pd.DataFrame({"BGGId": game_ids, "Name": [f"Synthetic Game {i:02d}" for i in range(n_games)],
                  "MinPlayers": rng.integers(1, 3, n_games)}).to_csv(path / "games.csv", index=False)
    mechanics.to_csv(path / "mechanics.csv", index=False)
    themes.to_csv(path / "themes.csv", index=False)
    return path
