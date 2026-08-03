'''Branch-value comparison against a reference ntuple (refcheck.compare_ntuples).

The per-step count gate asks how many models cleared each step. For the emcee chain
that is near-vacuous -- every model clears every step, so 54/54 holds no matter where
the walkers went. Since a fixed seed reproduces the chain exactly
(tests/test_emcee_determinism.py), the values themselves can be gated, which is what
actually pins the trajectory. upload_references.py has always published the reference
ntuple alongside the counts sidecar; nothing read it until now.
'''
import numpy as np
import pytest
import uproot

from Run3ModelGen.refcheck import VALUE_RTOL, compare_ntuples


BRANCHES = {"IN_M_1": [100.0, -250.5, 700.25],
            "SP_m_h": [125.09, 122.4, 128.8],
            "emcee_logp": [-1.5, -20.0, -0.25],
            "chi2": [3.0, 40.0, 0.5]}


def _write(path, data, tree="susy"):
    with uproot.recreate(path) as f:
        f[tree] = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
    return str(path)


@pytest.fixture
def ref(tmp_path):
    return _write(tmp_path / "ref.root", BRANCHES)


# ── the happy path ──────────────────────────────────────────────────────────────

def test_identical_ntuples_match(ref, tmp_path):
    ok, table = compare_ntuples(ref, _write(tmp_path / "p.root", BRANCHES))
    assert ok, table
    assert "all 4 branches match" in table


def test_last_ulp_noise_is_tolerated(ref, tmp_path):
    '''Deliberately NOT a bitwise gate: an exact comparison would fail on a one-ulp
    SPheno difference that means nothing physically. VALUE_RTOL is ~4 orders tighter
    than any meaningful drift.'''
    nudged = {k: [v * (1 + 1e-15) for v in vals] for k, vals in BRANCHES.items()}
    ok, _ = compare_ntuples(ref, _write(tmp_path / "p.root", nudged))
    assert ok


# ── the failures it must catch ──────────────────────────────────────────────────

def test_a_real_value_change_fails(ref, tmp_path):
    moved = dict(BRANCHES, SP_m_h=[125.09, 122.4, 124.0])     # one entry, 4% off
    ok, table = compare_ntuples(ref, _write(tmp_path / "p.root", moved))
    assert not ok
    assert "SP_m_h" in table and "1" in table


def test_a_drift_just_above_tolerance_fails(ref, tmp_path):
    '''The gate must actually bite near its threshold, not only on gross changes.'''
    shifted = dict(BRANCHES, chi2=[v * (1 + 1e-9) for v in BRANCHES["chi2"]])
    ok, _ = compare_ntuples(ref, _write(tmp_path / "p.root", shifted), rtol=VALUE_RTOL)
    assert not ok


def test_a_truncated_chain_is_named_as_such(ref, tmp_path):
    '''The one flake mode a value gate inherits from the chain: the sampling loop's
    wall-clock guard checkpoints early, so the ntuple has fewer entries. Report it as a
    truncation rather than as thousands of value mismatches.'''
    short = {k: v[:2] for k, v in BRANCHES.items()}
    ok, table = compare_ntuples(ref, _write(tmp_path / "p.root", short))
    assert not ok
    assert "TRUNCATED" in table
    assert "reference 3, produced 2" in table


def test_a_missing_branch_fails(ref, tmp_path):
    fewer = {k: v for k, v in BRANCHES.items() if k != "chi2"}
    ok, table = compare_ntuples(ref, _write(tmp_path / "p.root", fewer))
    assert not ok
    assert "chi2" in table and "MISSING" in table


def test_an_extra_branch_fails(ref, tmp_path):
    more = dict(BRANCHES, SP_m_A=[1.0, 2.0, 3.0])
    ok, table = compare_ntuples(ref, _write(tmp_path / "p.root", more))
    assert not ok
    assert "SP_m_A" in table and "NOT IN reference" in table


# ── NaN handling ────────────────────────────────────────────────────────────────

def test_nan_matches_nan(tmp_path):
    '''An unfilled observable is real data in this comparison, not noise: a chain whose
    blob came back nan must compare equal to a reference that also has nan there.'''
    withnan = dict(BRANCHES, SP_m_h=[125.09, float("nan"), 128.8])
    a = _write(tmp_path / "a.root", withnan)
    b = _write(tmp_path / "b.root", withnan)
    ok, _ = compare_ntuples(a, b)
    assert ok


def test_nan_versus_a_number_fails(ref, tmp_path):
    withnan = dict(BRANCHES, SP_m_h=[125.09, float("nan"), 128.8])
    ok, table = compare_ntuples(ref, _write(tmp_path / "p.root", withnan))
    assert not ok
    assert "SP_m_h" in table
