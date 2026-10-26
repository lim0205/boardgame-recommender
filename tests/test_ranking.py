import numpy as np
import pandas as pd
import pytest
from boardrec.ranking import PopularityRanker, BlendedRanker, _percentile_ranks
from boardrec.ranking_experiments import eligible_sample

class ToyModel:
    def predict(self, uid, game): return float(self.predict_many(uid, [game])[0])
    def predict_many(self, uid, ids): return np.array([{"b":9,"c":1,"d":7}.get(g, 5) for g in ids], dtype=float)


def test_popularity_excludes_train_rated_and_never_uses_val():
    train=pd.DataFrame({"user_id":["u","x","y","z","w"],"game_id":["a","b","b","c","b"],"rating":[8,9,9,7,8]})
    pop=PopularityRanker().fit(train,["a","b","c","d"])
    assert pop.recommend("u",n=2).game_id.tolist() == ["b","c"]
    assert "a" not in pop.recommend("u",n=4).game_id.tolist()
    assert pop.popularity_["d"] == 0


def test_blended_endpoints_and_prediction_scale():
    train=pd.DataFrame({"user_id":["u","x","y","z"],"game_id":["a","c","c","b"],"rating":[8,6,8,4]})
    pop=PopularityRanker().fit(train,["a","b","c","d"])
    assert BlendedRanker(ToyModel(),pop,0).recommend("u",1).game_id.iloc[0] == "c"
    assert BlendedRanker(ToyModel(),pop,1).recommend("u",1).game_id.iloc[0] == "b"
    assert BlendedRanker(ToyModel(),pop,.5).predict("u","d") == 7
    with pytest.raises(ValueError): BlendedRanker(ToyModel(),pop,1.4)


def test_ties_not_arbitrarily_promoted():
    np.testing.assert_allclose(_percentile_ranks(np.array([4,4,4])), [0.5]*3)
    np.testing.assert_allclose(_percentile_ranks(np.array([1,1,9])), [0.25,0.25,1])


def test_positive_user_sampling_is_reproducible():
    data=pd.DataFrame({"user_id":[str(i) for i in range(90)],
                       "game_id":["g"]*90,"rating":[8]*90})
    a,_,n=eligible_sample(data,max_users=8)
    b,_,_=eligible_sample(data,max_users=8)
    assert a==b and len(a)==8 and n==90


def test_balanced_selection_uses_validation_only():
    from boardrec.ranking_experiments import select_balanced_candidate
    rows = pd.DataFrame([
        {"feature_set":"both", "content_weight":0., "validation_hit_rate_at_10":.339,
         "distinct_recommended_games":54},
        {"feature_set":"both", "content_weight":.1, "validation_hit_rate_at_10":.322,
         "distinct_recommended_games":98},
        {"feature_set":"both", "content_weight":.2, "validation_hit_rate_at_10":.287,
         "distinct_recommended_games":130}
    ])
    assert select_balanced_candidate(rows,{"feature_set":"both", "content_weight":0.}) == {
        "feature_set":"both", "content_weight":.1}
