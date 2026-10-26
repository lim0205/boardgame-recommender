"""Fast BGG preparation must accept the actual BGG column layout."""
import pandas as pd
from boardrec.demo_data import generate_demo
from boardrec.bgg_fast import prepare_bgg_fast
from boardrec.preprocessing import PrepConfig, load_prepared


def test_fast_prepare_and_splits(tmp_path):
    raw = generate_demo(tmp_path / "raw")
    output = tmp_path / "processed"
    manifest = prepare_bgg_fast(
        raw / "user_ratings.csv", raw / "games.csv",
        raw / "mechanics.csv", raw / "themes.csv",
        output, PrepConfig(top_games=35, min_user_ratings=8,
                           max_users=20, seed=42))
    train, val, test, games, mechanics, themes = load_prepared(output)
    assert manifest["n_users"] <= 20
    assert manifest["n_ratings"] == len(train) + len(val) + len(test)
    assert manifest["n_raw_ratings"] > manifest["n_ratings"]
    assert not pd.concat([train, val, test]).duplicated(["user_id", "game_id"]).any()
    assert set(train.game_id).issubset(set(games.game_id))
    assert set(games.game_id) == set(mechanics.game_id) == set(themes.game_id)
