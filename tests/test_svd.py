import numpy as np
import pandas as pd
import pytest
from boardrec.svd import SVDRecommender
from boardrec.evaluation import evaluate


def fixture():
    train = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
        "game_id": ["A", "B", "C", "A", "C", "D", "B", "C", "D"],
        "rating": [9, 5, 1, 8, 2, 5, 4, 7, 9],
    })
    catalog = pd.DataFrame({"game_id": ["A", "B", "C", "D", "E"]})
    return train, catalog


def test_fit_predict_and_recommend():
    train, catalog = fixture()
    model = SVDRecommender(n_factors=2, seed=3).fit(train, catalog)
    arr = model.predict_many("u1", ["D", "E"])
    assert len(arr) == 2
    assert np.isfinite(arr).all()
    assert np.all((1 <= arr) & (arr <= 10))
    assert model.predict("u1", "D") == pytest.approx(arr[0])
    assert model.recommend("u1", n=10).game_id.tolist() == sorted(set(catalog.game_id) - {"A", "B", "C"}, key=lambda g: (-model.predict("u1", g), g))


def test_unknown_user_and_unrated_catalog_game():
    train, games = fixture()
    model = SVDRecommender(n_factors=50).fit(train, games)
    assert model.effective_factors_ == 2  # min(n_users,n_items)-1
    assert model.predict("brand-new", "E") == pytest.approx(train.rating.mean())
    assert model.predict("u1", "E") == pytest.approx(train[train.user_id == "u1"].rating.mean())
    assert "A" not in set(model.recommend("u1").game_id)
    assert len(model.recommend("brand-new")) == len(games)


def test_evaluation_compatibility_and_reproducibility():
    train, games = fixture()
    m1 = SVDRecommender(n_factors=2, seed=42).fit(train, games)
    m2 = SVDRecommender(n_factors=2, seed=42).fit(train, games)
    np.testing.assert_allclose(m1.predict_many("u2", ["A", "B", "C", "D"]),
                               m2.predict_many("u2", ["A", "B", "C", "D"]), atol=1e-10)
    validation = pd.DataFrame({"user_id": ["u1", "u2", "u3"],
                               "game_id": ["D", "B", "A"], "rating": [9., 8., 8.]})
    report = evaluate(m1, validation, max_hit_users=None)
    assert "rmse" in report and "mae" in report
    assert "hit_rate@10" in report
    assert report["rmse"] >= 0


def test_validation_errors():
    train, games = fixture()
    with pytest.raises(ValueError):
        SVDRecommender(n_factors=0)
    with pytest.raises(ValueError, match="Duplicate"):
        SVDRecommender().fit(pd.concat([train, train.iloc[:1]]), games)
    with pytest.raises(ValueError, match="missing"):
        SVDRecommender().fit(train, pd.DataFrame({"game_id": ["A"]}))
    with pytest.raises(ValueError, match="range"):
        bad = train.copy()
        bad.loc[0, "rating"] = 11
        SVDRecommender().fit(bad, games)
