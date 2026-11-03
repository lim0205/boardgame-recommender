import numpy as np
import pandas as pd
import pytest

from boardrec.content import ContentBasedRecommender
from boardrec.demo_data import generate_demo
from boardrec.evaluation import evaluate
from boardrec.hybrid import CachedPredictions, HybridRecommender
from boardrec.hybrid_experiments import run_hybrid_experiments
from boardrec.memory_cf import MemoryCFRecommender
from boardrec.preprocessing import PrepConfig, load_prepared, prepare
from boardrec.svd import SVDRecommender


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hybrid")
    raw = generate_demo(tmp / "raw")
    out = tmp / "prepared"
    prepare(raw / "user_ratings.csv", raw / "games.csv", raw / "mechanics.csv",
            raw / "themes.csv", out, PrepConfig(top_games=60, min_user_ratings=8, max_users=20))
    return out


def _models(prepared):
    train, val, _, games, mech, themes = load_prepared(prepared)
    content = ContentBasedRecommender().fit(train, games, mech, themes)
    svd = SVDRecommender(n_factors=3).fit(train, games)
    return train, val, content, svd


def test_alpha_endpoints_and_blend(prepared):
    train, val, content, svd = _models(prepared)
    user = train.user_id.iloc[0]
    games = list(content.game_ids_[:8])
    c, s = content.predict_many(user, games), svd.predict_many(user, games)
    np.testing.assert_allclose(HybridRecommender(content, svd, 1.0, train).predict_many(user, games), c)
    np.testing.assert_allclose(HybridRecommender(content, svd, 0.0, train).predict_many(user, games), s)
    np.testing.assert_allclose(HybridRecommender(content, svd, 0.3, train).predict_many(user, games),
                               0.3 * c + 0.7 * s)


def test_recommend_excludes_train_and_evaluates(prepared):
    train, val, content, svd = _models(prepared)
    model = HybridRecommender(content, MemoryCFRecommender(k=5).fit(train), 0.5, train)
    user = train.user_id.iloc[0]
    seen = set(train.loc[train.user_id == user, "game_id"])
    assert not seen & set(model.recommend(user, n=10).game_id)
    report = evaluate(model, val, max_hit_users=5)
    assert 0 <= report["hit_rate@10"] <= 1 and report["rmse"] > 0


def test_cached_predictions_match(prepared):
    train, _, content, _ = _models(prepared)
    cached = CachedPredictions(content)
    user = train.user_id.iloc[1]
    games = list(content.game_ids_[:10])
    np.testing.assert_allclose(cached.predict_many(user, games[::-1]),
                               content.predict_many(user, games[::-1]))
    np.testing.assert_allclose(cached.predict_many(user, games), content.predict_many(user, games))


def test_invalid_alpha(prepared):
    train, _, content, svd = _models(prepared)
    with pytest.raises(ValueError):
        HybridRecommender(content, svd, 1.5, train)


def test_hybrid_experiment_validation_only(prepared, tmp_path):
    kwargs = dict(alphas=(0.0, 0.5, 1.0), svd_params={"n_factors": 3},
                  memory_cf_params={"k": 5}, max_hit_users=5)
    first = run_hybrid_experiments(prepared, tmp_path / "r1", **kwargs)
    assert first["best"]["cf_model"] in {"svd", "memory_cf"}
    assert first["best"]["alpha"] in (0.0, 0.5, 1.0)
    table = pd.read_csv(tmp_path / "r1" / "validation_metrics.csv")
    assert len(table) == 6
    test_path = prepared / "test.csv"
    original = pd.read_csv(test_path)
    try:
        original.assign(rating=1.0).to_csv(test_path, index=False)
        assert run_hybrid_experiments(prepared, tmp_path / "r2", **kwargs) == first
    finally:
        original.to_csv(test_path, index=False)


def test_missing_train_fails_loudly(prepared):
    _, _, content, svd = _models(prepared)
    HybridRecommender(content, svd, 0.5)  # content profile is usable
    with pytest.raises(ValueError, match="train"):
        HybridRecommender(CachedPredictions(content), svd, 0.5)
