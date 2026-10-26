from pathlib import Path
import pandas as pd
import pytest

from boardrec.preprocessing import PrepConfig, prepare, load_prepared, split_by_user
from boardrec.demo_data import generate_demo


def test_per_user_splits_are_disjoint_and_repeatable():
    rows = [(str(u), str(i), float((u+i) % 10 + 1)) for u in range(3) for i in range(30)]
    df = pd.DataFrame(rows, columns=["user_id", "game_id", "rating"])
    cfg = PrepConfig(min_user_ratings=20, max_users=None, seed=7)
    a, b = split_by_user(df, cfg), split_by_user(df, cfg)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(a[name], b[name])
    sets = [set(map(tuple, x[["user_id", "game_id"]].to_numpy())) for x in a.values()]
    assert len(set.union(*sets)) == len(df)
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
    assert all(s.user_id.nunique() == 3 for s in a.values())


def test_chunked_preparation_end_to_end(tmp_path):
    raw = generate_demo(tmp_path / "raw")
    cfg = PrepConfig(top_games=30, min_user_ratings=5, max_users=12, chunksize=41, seed=42)
    prepared = tmp_path / "processed"
    result = prepare(raw / "user_ratings.csv", raw / "games.csv", raw / "mechanics.csv",
                     raw / "themes.csv", prepared, cfg)
    train, val, test, games, mechanics, themes = load_prepared(prepared)
    assert result["n_users"] <= 12
    assert result["n_games"] <= 30
    assert result["n_ratings"] == len(train) + len(val) + len(test)
    assert set(train.columns) == {"user_id", "game_id", "rating"}
    assert train.user_id.dtype == object
    assert set(train.game_id).issubset(set(games.game_id))
    assert set(games.game_id) == set(mechanics.game_id) == set(themes.game_id)
    assert not pd.concat([train, val, test]).duplicated(["user_id", "game_id"]).any()
    assert (prepared / "manifest.json").exists()


def test_validation_rejects_bad_config():
    with pytest.raises(ValueError):
        PrepConfig(min_user_ratings=2)
