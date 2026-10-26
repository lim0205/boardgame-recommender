import numpy as np
import pandas as pd
import pytest

from boardrec.content import ContentBasedRecommender


def sample_frames():
    train = pd.DataFrame({"user_id": ["1", "1", "2", "2"],
                          "game_id": ["A", "B", "A", "B"],
                          "rating": [9., 3., 2., 8.]})
    games = pd.DataFrame({"BGGId": ["A", "B", "C", "D"],
                          "Name": ["A game", "B game", "C game", "D game"]})
    mechanics = pd.DataFrame({"BGGId": ["A", "B", "C", "D"],
                              "dice": [1, 0, 1, 0], "cards": [0, 1, 0, 0]})
    themes = pd.DataFrame({"BGGId": ["A", "B", "C", "D"],
                           "fantasy": [1, 0, 1, 0]})
    return train, games, mechanics, themes


def test_predict_and_recommend_and_unrated_only():
    train, games, mechanics, themes = sample_frames()
    model = ContentBasedRecommender(feature_set="both").fit(train, games, mechanics, themes)
    assert np.isclose(model.predict("1", "C"), 9.0)
    assert np.isclose(model.predict("1", "D"), 6.0)  # No shared features -> mean
    assert np.isclose(model.predict("new-user", "C"), 5.5)  # Not personalized
    ranked = model.recommend("1", n=10)
    assert set(ranked.game_id) == {"C", "D"}
    assert ranked.iloc[0].game_id == "C"
    assert np.allclose(model.predict_many("1", ["C", "D"]), [9, 6])


def test_validation_failures():
    train, games, mechanics, themes = sample_frames()
    with pytest.raises(ValueError):
        ContentBasedRecommender(feature_set="unknown")
    model = ContentBasedRecommender()
    with pytest.raises(RuntimeError):
        model.predict("1", "A")
    model.fit(train, games, mechanics, themes)
    with pytest.raises(KeyError):
        model.predict("1", "UNKNOWN")
