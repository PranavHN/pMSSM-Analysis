'''Candidate generation in the free box and the labelled-set bookkeeping (torch-free).'''
import numpy as np
import pytest

from Run3ModelGen.active_learning import data as ald

PARAMS = {"tanb": [1, 60], "M_1": [-2000, 2000], "M_3": [4000, 4000],
          "M_2": [[-3000, -1000], [1000, 3000]], "mA": [2000, 2000]}


def test_free_and_fixed_follow_the_config_order():
    assert ald.free_parameters(PARAMS) == ["tanb", "M_1", "M_2"]
    assert ald.fixed_parameters(PARAMS) == {"M_3": 4000.0, "mA": 2000.0}
    lo, hi = ald.free_box(PARAMS)
    assert list(lo) == [1, -2000, -3000] and list(hi) == [60, 2000, 3000]


@pytest.mark.parametrize("method", ["lhs", "uniform"])
def test_candidates_fill_the_box_and_respect_gaps(method):
    X = ald.generate_candidates(PARAMS, 4000, method, seed=3)
    assert X.shape == (4000, 3)
    assert X[:, 0].min() >= 1 and X[:, 0].max() <= 60
    assert X[:, 1].min() >= -2000 and X[:, 1].max() <= 2000
    m2 = X[:, 2]
    assert ((m2 >= -3000) & (m2 <= -1000) | (m2 >= 1000) & (m2 <= 3000)).all(), "gap must be empty"
    assert abs((m2 > 0).mean() - 0.5) < 0.05, "sub-ranges are weighted by width"


def test_lhs_is_stratified_per_dimension():
    n = 1000
    X = ald.generate_candidates(PARAMS, n, "lhs", seed=1)
    u = (X[:, 1] + 2000) / 4000
    counts = np.histogram(u, bins=n, range=(0, 1))[0]
    assert counts.max() == 1, "one point per stratum"


def test_candidates_are_seeded():
    a = ald.generate_candidates(PARAMS, 50, "lhs", seed=11)
    b = ald.generate_candidates(PARAMS, 50, "lhs", seed=11)
    c = ald.generate_candidates(PARAMS, 50, "lhs", seed=12)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_to_points_fills_the_fixed_parameters():
    X = np.array([[10.0, 100.0, 1500.0], [20.0, -100.0, -1500.0]])
    pts = ald.to_points(X, PARAMS)
    assert set(pts) == set(PARAMS)
    assert list(pts["tanb"]) == [10.0, 20.0] and list(pts["M_2"]) == [1500.0, -1500.0]
    assert list(pts["M_3"]) == [4000.0, 4000.0] and list(pts["mA"]) == [2000.0, 2000.0]


def test_labelled_set_bookkeeping_and_duplicates():
    s = ald.LabelledSet(free_names=["a", "b"])
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    s.append(X, y=[0.1, -1.0, 0.12], valid=[True, False, True], hit=[False, False, True],
             role=[0, 0, 1], iteration=0, attempt=0, index=[0, 1, 2])
    assert len(s) == 3
    assert list(s.train_rows()) == [0] and list(s.val_rows()) == [2]
    assert np.isnan(s.pred).all()
    dup = s.is_duplicate(np.array([[3.0, 4.0], [7.0, 8.0], [7.0, 8.0]]))
    assert list(dup) == [True, False, True]
    s.append(np.array([[7.0, 8.0]]), y=[0.2], valid=[True], hit=[False], role=[0], iteration=1,
             attempt=0, index=[0], pred=[0.5], score=[1.5])
    assert list(s.iteration) == [0, 0, 0, 1] and s.score[-1] == 1.5
    back = ald.LabelledSet.from_dict(s.to_dict())
    assert len(back) == 4 and back.free_names == ["a", "b"] and np.array_equal(back.X, s.X)


def test_split_roles_gives_the_validation_fraction():
    rng = np.random.default_rng(0)
    role = ald.split_roles(100, 0.2, rng)
    assert (role == ald.ROLE_VAL).sum() == 20
    assert (ald.split_roles(1, 0.2, rng) == ald.ROLE_TRAIN).all(), "a single point trains"
    assert (ald.split_roles(2, 0.2, rng) == ald.ROLE_VAL).sum() == 1, "at least one validation point"
