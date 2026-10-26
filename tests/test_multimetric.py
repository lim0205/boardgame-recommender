import math
from scipy import sparse
from boardrec.multimetric import _per_user_metrics, _item_diversity


def test_observed_ndcg_prioritizes_high_rating():
    actual = {'b': 10, 'a': 8}
    good = _per_user_metrics(['b', 'a'], actual, k=2)
    bad = _per_user_metrics(['a', 'b'], actual, k=2)
    assert good['ndcg'] == 1.0
    assert bad['ndcg'] < good['ndcg']
    assert good['recall'] == 1 and good['hit'] == 1


def test_observed_recall_distinct_from_hit():
    m = _per_user_metrics(['a', 'c'], {'a': 8, 'b': 10}, k=2)
    assert m['hit'] == 1
    assert m['recall'] == 0.5
    assert m['ndcg'] < 1


def test_no_observed_positive_returns_zero():
    assert _per_user_metrics(['x'], {}, k=1) == {'hit': 0, 'recall': 0.0, 'ndcg': 0.0}


def test_intra_list_diversity():
    vectors = sparse.csr_matrix([[1., 0.], [0., 1.], [1., 0.]])
    assert math.isclose(_item_diversity([0,1], vectors), 1.0)
    assert math.isclose(_item_diversity([0,2], vectors), 0.0)
