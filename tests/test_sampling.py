'''Sampler behavior and the RNG-determinism property the CI references rest on.

test_sampling_fixture.py guards byte-compatibility against the pre-refactor
code; these tests guard the *properties* — bounds, shapes, overrides, boundary
handling — so a change that happens to keep the golden file valid still cannot
break the physics.
'''

import csv

import numpy as np
import pytest

from Run3ModelGen import sampling


PARAMS = {
    "tanb": [1, 60],
    "M_1": [-2000, 2000],
    "M_2": [2000, 2000],          # fixed (lo == hi)
    "M_3": [[-3000, -1000], [1000, 3000]],   # gapped/interrupted range
}


# ── flat ───────────────────────────────────────────────────────────────────

def test_flat_shapes_and_bounds():
    np.random.seed(1)
    pts = sampling.sample_flat(PARAMS, 50)
    assert set(pts) == set(PARAMS)
    for key in PARAMS:
        assert len(pts[key]) == 50
    assert np.all((pts["tanb"] >= 1) & (pts["tanb"] <= 60))
    assert np.all(pts["M_2"] == 2000)


def test_flat_is_reproducible_for_a_given_seed():
    np.random.seed(42)
    a = sampling.sample_flat(PARAMS, 20)
    np.random.seed(42)
    b = sampling.sample_flat(PARAMS, 20)
    for key in a:
        assert np.array_equal(a[key], b[key])


def test_flat_differs_between_seeds():
    np.random.seed(1)
    a = sampling.sample_flat(PARAMS, 20)
    np.random.seed(2)
    b = sampling.sample_flat(PARAMS, 20)
    assert not np.array_equal(a["tanb"], b["tanb"])


def test_flat_respects_gaps_in_interrupted_ranges():
    np.random.seed(3)
    pts = sampling.sample_flat(PARAMS, 200)
    m3 = pts["M_3"]
    assert len(m3) == 200
    in_lower = (m3 >= -3000) & (m3 <= -1000)
    in_upper = (m3 >= 1000) & (m3 <= 3000)
    assert np.all(in_lower | in_upper), "no point may fall inside the gap"
    assert in_lower.any() and in_upper.any(), "both subranges should be populated"


def test_flat_rejects_a_malformed_subrange():
    with pytest.raises(ValueError):
        sampling.sample_flat({"x": [[0, 1], 5]}, 10)


# ── fixed ──────────────────────────────────────────────────────────────────

def test_fixed_uses_the_supplied_values_verbatim():
    pts = sampling.sample_fixed({"tanb": [10, 20, 30], "M_1": [-1, 0, 1]})
    assert np.array_equal(pts["tanb"], np.array([10, 20, 30]))
    assert np.array_equal(pts["M_1"], np.array([-1, 0, 1]))


# ── grid ───────────────────────────────────────────────────────────────────

def test_grid_is_the_cartesian_product_and_overrides_num_models():
    np.random.seed(1)
    grid = {"M_1": [-500, 0, 500], "tanb": [10, 50]}
    pts, n = sampling.sample_grid(PARAMS, grid)
    assert n == 6                      # 3 x 2, independent of num_models
    for key in PARAMS:
        assert len(pts[key]) == 6
    assert sorted(set(pts["M_1"])) == [-500.0, 0.0, 500.0]
    assert sorted(set(pts["tanb"])) == [10.0, 50.0]
    # every combination appears exactly once
    combos = sorted(zip(pts["M_1"], pts["tanb"]))
    assert len(set(combos)) == 6


def test_grid_randomises_non_axis_parameters_within_bounds():
    np.random.seed(2)
    pts, n = sampling.sample_grid(PARAMS, {"tanb": [10, 50]})
    assert np.all(pts["M_2"] == 2000), "fixed parameters stay fixed"
    assert np.all((pts["M_1"] >= -2000) & (pts["M_1"] <= 2000))
    m3 = pts["M_3"]
    assert np.all(((m3 >= -3000) & (m3 <= -1000)) | ((m3 >= 1000) & (m3 <= 3000)))


def test_grid_requires_a_grid_block():
    with pytest.raises(ValueError):
        sampling.sample_grid(PARAMS, {})


def test_grid_refuses_to_exceed_the_model_cap():
    big = {f"p{i}": list(range(10)) for i in range(5)}   # 10^5 points
    with pytest.raises(SystemExit):
        sampling.sample_grid(PARAMS, big)


# ── importance ─────────────────────────────────────────────────────────────

def _seed_csv(tmp_path, rows):
    path = tmp_path / "seeds.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_importance_multiplies_seeds_by_perturbations(tmp_path):
    np.random.seed(1)
    seeds = _seed_csv(tmp_path, [
        {"tanb": 10, "M_1": 100, "M_2": 2000, "M_3": 2000},
        {"tanb": 20, "M_1": -100, "M_2": 2000, "M_3": -2000},
    ])
    pts, n = sampling.sample_importance(PARAMS, {"seed_file": seeds, "perturbations_per_seed": 5})
    assert n == 10
    for key in PARAMS:
        assert len(pts[key]) == 10


def test_importance_stays_near_its_seed_and_inside_bounds(tmp_path):
    np.random.seed(7)
    seeds = _seed_csv(tmp_path, [{"tanb": 30, "M_1": 0, "M_2": 2000, "M_3": 2000}])
    pts, _ = sampling.sample_importance(
        PARAMS, {"seed_file": seeds, "perturbations_per_seed": 40,
                 "proposal_widths": {"tanb": 0.5, "M_1": 10.0, "M_2": 0.0, "M_3": 0.0}})
    assert np.all(np.abs(pts["tanb"] - 30) < 5), "narrow width must stay near the seed"
    assert np.all((pts["tanb"] >= 1) & (pts["tanb"] <= 60))
    assert np.all(pts["M_2"] == 2000), "zero width keeps the parameter at its seed"


def test_importance_needs_an_existing_seed_file():
    with pytest.raises(FileNotFoundError):
        sampling.sample_importance(PARAMS, {"seed_file": "/nope/seeds.csv"})


def test_importance_requires_a_block():
    with pytest.raises(ValueError):
        sampling.sample_importance(PARAMS, {})


# ── proposal helpers (used by importance sampling) ─────────────────────────

def test_default_proposal_width_is_five_percent_of_the_range():
    widths = sampling.build_proposal_widths({"x": [0, 100]}, {})
    assert widths["x"] == pytest.approx(5.0)


def test_configured_proposal_width_wins():
    widths = sampling.build_proposal_widths({"x": [0, 100]}, {"proposal_widths": {"x": 13.0}})
    assert widths["x"] == 13.0


def test_gapped_range_width_uses_the_summed_subranges():
    widths = sampling.build_proposal_widths({"x": [[0, 100], [200, 300]]}, {})
    assert widths["x"] == pytest.approx(10.0)


@pytest.mark.parametrize("value, expected", [
    (5.0, 5.0),      # inside
    (-2.0, 2.0),     # below -> reflected
    (12.0, 8.0),     # above -> reflected
])
def test_reflect_maps_into_the_interval(value, expected):
    assert sampling.reflect(value, 0.0, 10.0) == pytest.approx(expected)


def test_reflect_handles_a_degenerate_interval():
    assert sampling.reflect(123.0, 5.0, 5.0) == 5.0


def test_reflect_converges_for_far_out_values():
    for v in (1e4, -1e4, 250.0):
        assert 0.0 <= sampling.reflect(v, 0.0, 10.0) <= 10.0


def test_propose_keeps_zero_width_parameters_fixed():
    out = sampling.propose({"x": 3.0}, {"x": 0.0}, {"x": [0, 10]})
    assert out["x"] == 3.0


def test_propose_rejects_moves_into_a_gap():
    np.random.seed(5)
    params = {"x": [[0, 10], [90, 100]]}
    for _ in range(50):
        out = sampling.propose({"x": 5.0}, {"x": 30.0}, params)
        assert (0 <= out["x"] <= 10) or (90 <= out["x"] <= 100)


def test_sample_initial_point_is_inside_bounds():
    np.random.seed(9)
    for _ in range(20):
        point = sampling.sample_initial_point(PARAMS)
        assert 1 <= point["tanb"] <= 60
        assert point["M_2"] == 2000
        assert (-3000 <= point["M_3"] <= -1000) or (1000 <= point["M_3"] <= 3000)
