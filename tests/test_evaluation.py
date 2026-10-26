import math
import pandas as pd
import pytest

from boardrec.adapters import PredictAdapter
from boardrec.evaluation import evaluate, hit_rate_at_k, rating_metrics


def fixture_model():
    train = pd.DataFrame({"user_id": ["u", "u", "v"],
                          "game_id": ["A", "B", "A"], "rating": [9., 3., 5.]})
    scores = {("u", "C"): 9., ("u", "D"): 7., ("u", "E"): 4.,
              ("v", "C"): 2., ("v", "D"): 6., ("v", "E"): 9.}
    return PredictAdapter(lambda u, g: scores.get((u, g), 5.0),
                          train, ["A", "B", "C", "D", "E"])


def test_rating_and_hit_metrics():
    model = fixture_model()
    holdout = pd.DataFrame({"user_id": ["u", "u", "v"],
                            "game_id": ["C", "D", "E"], "rating": [8., 5., 9.]})
    result = evaluate(model, holdout, top_k=1, max_hit_users=None)
    assert math.isclose(result["rmse"], math.sqrt(5/3))
    assert math.isclose(result["mae"], 1.0)
    assert result["hit_rate@1"] == 1.0
    assert result["eligible_users"] == 2
    assert set(model.recommend("u", n=3).game_id) == {"C", "D", "E"}


def test_empty_positive_case_and_empty_holdout():
    model = fixture_model()
    holdout = pd.DataFrame({"user_id": ["u"], "game_id": ["D"], "rating": [4.]})
    result = hit_rate_at_k(model, holdout)
    assert math.isnan(result["hit_rate@10"])
    assert result["eligible_users"] == 0
    empty = holdout.iloc[:0]
    assert math.isnan(rating_metrics(model, empty)["rmse"])


def test_adapter_rejects_invalid_output():
    model = PredictAdapter(lambda u, g: float("nan"),
                           pd.DataFrame({"user_id": ["u"], "game_id": ["A"], "rating": [1]}),
                           ["A", "B"])
    with pytest.raises(ValueError):
        model.predict("u", "B")
