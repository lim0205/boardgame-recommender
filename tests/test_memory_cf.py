import numpy as np
import pandas as pd
import pytest

from boardrec.demo_data import generate_demo
from boardrec.evaluation import evaluate
from boardrec.memory_cf import MemoryCFRecommender
from boardrec.memory_cf_experiments import run_memory_cf_experiments
from boardrec.preprocessing import PrepConfig, prepare


def fixture():
    train = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3", "u4", "u4", "u4"],
        "game_id": ["A", "B", "C", "A", "B", "D", "B", "C", "D", "A", "C", "D"],
        "rating": [9, 8, 2, 9, 8, 7, 3, 9, 4, 8, 3, 6],
    })
    catalog = pd.DataFrame({"game_id": ["A", "B", "C", "D", "E"]})
    return train, catalog


@pytest.mark.parametrize("kind", ["user", "item"])
@pytest.mark.parametrize("sim", ["cosine", "pearson"])
def test_api_ranges_and_exclusion(kind, sim):
    train, catalog = fixture()
    model = MemoryCFRecommender(kind=kind, similarity=sim, k=2, min_support=1).fit(train, catalog)
    preds = model.predict_many("u1", ["D", "E"])
    assert preds.shape == (2,) and np.isfinite(preds).all()
    assert np.all((preds >= 1) & (preds <= 10))
    assert model.predict("u1", "D") == pytest.approx(preds[0])
    rec = model.recommend("u1", n=10)
    assert set(rec.game_id) == {"D", "E"}
    assert list(rec.score) == sorted(rec.score, reverse=True)


def test_baseline_fallbacks():
    train, catalog = fixture()
    model = MemoryCFRecommender(kind="item", min_support=1).fit(train, catalog)
    # Game E has no train ratings: no item bias, no neighbours -> user mean.
    u1_mean = train[train.user_id == "u1"].rating.mean()
    assert model.predict("u1", "E") == pytest.approx(u1_mean)
    # Unknown user -> global mean (+ item bias 0 for E).
    assert model.predict("new-user", "E") == pytest.approx(train.rating.mean())
    assert len(model.recommend("new-user")) == len(catalog)


def test_user_based_neighbour_signal():
    # u2 is u1's closest neighbour and loved D; u1 should be predicted above baseline for D.
    train, catalog = fixture()
    model = MemoryCFRecommender(kind="user", similarity="pearson", k=1, min_support=2,
                                shrinkage=0).fit(train, catalog)
    base = model.user_means_[model.user_positions_["u1"]] + model.item_bias_[model.game_positions_["D"]]
    assert model.similarity_[model.user_positions_["u1"], model.user_positions_["u2"]] > 0
    assert model.predict("u1", "D") > base


def test_similarity_symmetry_and_support():
    train, catalog = fixture()
    model = MemoryCFRecommender(kind="item", similarity="cosine", min_support=10).fit(train, catalog)
    assert np.allclose(model.similarity_, 0)  # nobody has 10 co-ratings
    model = MemoryCFRecommender(kind="user", similarity="pearson", min_support=1).fit(train, catalog)
    assert np.allclose(model.similarity_, model.similarity_.T, atol=1e-6)
    assert np.allclose(np.diag(model.similarity_), 0)


def test_validation_errors():
    train, catalog = fixture()
    with pytest.raises(ValueError):
        MemoryCFRecommender(kind="both")
    with pytest.raises(ValueError):
        MemoryCFRecommender(similarity="jaccard")
    with pytest.raises(ValueError):
        MemoryCFRecommender(k=0)
    with pytest.raises(ValueError, match="Duplicate"):
        MemoryCFRecommender().fit(pd.concat([train, train.iloc[:1]]), catalog)
    with pytest.raises(KeyError):
        MemoryCFRecommender().fit(train, catalog).predict("u1", "ZZZ")


def test_evaluator_compatibility():
    train, catalog = fixture()
    model = MemoryCFRecommender(min_support=1).fit(train, catalog)
    val = pd.DataFrame({"user_id": ["u1", "u3"], "game_id": ["D", "A"], "rating": [9.0, 8.0]})
    report = evaluate(model, val, max_hit_users=None)
    assert report["rmse"] >= 0 and report["candidate_catalog_size"] == 5


def test_experiment_is_validation_only(tmp_path):
    raw = generate_demo(tmp_path / "raw")
    prepared = tmp_path / "prepared"
    prepare(raw / "user_ratings.csv", raw / "games.csv", raw / "mechanics.csv",
            raw / "themes.csv", prepared, PrepConfig(top_games=60, min_user_ratings=8, max_users=20))
    kwargs = dict(ks=(5, 10), max_hit_users=5)
    first = run_memory_cf_experiments(prepared, tmp_path / "r1", **kwargs)
    assert first["best_hyperparameters"]["k"] in (5, 10)
    for name in ("validation_metrics.csv", "results.json", "validation_metrics.svg"):
        assert (tmp_path / "r1" / name).stat().st_size > 0
    test = pd.read_csv(prepared / "test.csv")
    test["rating"] = 1.0
    test.to_csv(prepared / "test.csv", index=False)
    assert run_memory_cf_experiments(prepared, tmp_path / "r2", **kwargs) == first
