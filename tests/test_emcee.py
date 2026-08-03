'''Unit tests for the emcee MCMC layer (`prior: mcmc`).

Pure-layer only, so they run in seconds with no physics tools and no cluster:
the sampler's helpers (`build_moves`, `resolve_burnin`, `_log_prior`), the
diagnostics (the `autocorr_time` nan-guard and the convergence verdict), the
shared HDF5 -> ROOT writer's branch layout + collision guard, and the emcee
config-schema validation. A tiny in-process emcee run (Gaussian toy target)
backs the ntuple tests — the real sampler is never invoked.
'''
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from Run3ModelGen import emcee_sampler as es
from Run3ModelGen import emcee_ntuple as en
from Run3ModelGen.modelgen import ModelGenerator


# The diagnostics live in a stand-alone script (deliberately not on the package
# import path), so load them by file path the way the other CLI tests do.
def _load_diag(repo):
    path = repo / "source" / "Run3ModelGen" / "scripts" / "emcee_diagnostics.py"
    spec = importlib.util.spec_from_file_location("emcee_diagnostics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def diag(repo):
    return _load_diag(repo)


# ── build_moves ───────────────────────────────────────────────────────────────

def test_build_moves_none_is_default():
    assert es.build_moves(None) is None
    assert es.build_moves([]) is None


def test_build_moves_valid():
    import emcee
    moves = es.build_moves([{"type": "StretchMove", "kwargs": {"a": 4.0}, "weight": 2.0}])
    assert len(moves) == 1
    mv, w = moves[0]
    assert isinstance(mv, emcee.moves.StretchMove)
    assert w == 2.0


def test_build_moves_unknown_type_raises():
    with pytest.raises(ValueError):
        es.build_moves([{"type": "NotARealMove"}])


def test_build_moves_nonpositive_weight_raises():
    with pytest.raises(ValueError):
        es.build_moves([{"type": "StretchMove", "weight": 0}])


# ── resolve_burnin (single source of truth) ─────────────────────────────────────

@pytest.mark.parametrize("max_tau,n_iter,floor,factor,expected", [
    (10.0, 1000, 0, 3.0, 30),          # ceil(factor*tau)
    (10.0, 1000, 100, 3.0, 100),       # absolute floor wins
    (1000.0, 100, 0, 3.0, 50),         # capped at n_iter // 2
    (float("nan"), 1000, 5, 3.0, 5),   # non-finite tau -> floor only
    (float("inf"), 1000, 0, 3.0, 0),   # inf tau is not finite -> 0
])
def test_resolve_burnin(max_tau, n_iter, floor, factor, expected):
    assert en.resolve_burnin(max_tau, n_iter, floor, factor) == expected


def test_resolve_burnin_diagnostics_delegates(diag):
    # The diagnostics copy must delegate to the shared one so the sampler's ntuple
    # discard and the diagnostics can never drift apart.
    for args in [(5.0, 500, 0, 3.0), (12.0, 80, 10, 2.0), (float("nan"), 300, 7, 3.0)]:
        assert diag.resolve_burnin(*args) == en.resolve_burnin(*args)


# ── _log_prior ──────────────────────────────────────────────────────────────────

def _mg_stub(free_names, parameters):
    return SimpleNamespace(free_names=free_names, parameters=parameters)


def test_log_prior_in_bounds_is_zero():
    mg = _mg_stub(["a", "b"], {"a": [0, 10], "b": [0, 10]})
    assert es._log_prior(mg, [5, 5]) == 0.0


def test_log_prior_out_of_bounds_is_neg_inf():
    mg = _mg_stub(["a"], {"a": [0, 10]})
    assert es._log_prior(mg, [11]) == -np.inf
    assert es._log_prior(mg, [-1]) == -np.inf


def test_log_prior_interrupted_ranges():
    # A list of sub-ranges (interrupted bounds): valid inside ANY sub-range.
    mg = _mg_stub(["a"], {"a": [[0, 1], [5, 6]]})
    assert es._log_prior(mg, [0.5]) == 0.0
    assert es._log_prior(mg, [5.5]) == 0.0
    assert es._log_prior(mg, [3.0]) == -np.inf


# ── autocorr_time nan-guard ─────────────────────────────────────────────────────

def test_autocorr_time_short_chain_not_ok(diag):
    rng = np.random.default_rng(0)
    chain = rng.standard_normal((20, 8, 2))   # N=20, far below 50*tau -> unreliable
    _tau, ok = diag.autocorr_time(chain, tol=50)
    assert not ok


def test_autocorr_time_constant_chain_nan_guard(diag):
    # A stuck / degenerate chain yields a non-finite tau; integrated_time does NOT
    # raise for it, so ok must still be False (regression: it used to report ok=True).
    chain = np.ones((200, 8, 2))
    tau, ok = diag.autocorr_time(chain, tol=50)
    assert not ok
    assert not np.all(np.isfinite(tau))


# ── convergence verdict ─────────────────────────────────────────────────────────

def _run_verdict(diag, rhat, ess, n50, acc, max_tau):
    diag._REPORT.clear()
    rhat_res = None if rhat is None else {"worst_rhat": rhat, "min_ess_bulk": ess}
    diag._emit_convergence_verdict(rhat_res, n50, acc, max_tau)
    return "\n".join(diag._REPORT)


def test_verdict_converged(diag):
    txt = _run_verdict(diag, 1.01, 1000, [True, True], [0.3], 20.0)
    assert "CONVERGED" in txt and "NOT CONVERGED" not in txt


def test_verdict_not_converged_high_rhat(diag):
    txt = _run_verdict(diag, 1.5, 1000, [True], [0.3], 20.0)
    assert "NOT CONVERGED" in txt


def test_verdict_not_converged_short_chain(diag):
    txt = _run_verdict(diag, 1.01, 1000, [False], [0.3], 20.0)
    assert "NOT CONVERGED" in txt


def test_verdict_inconclusive_degenerate(diag):
    # nan tau + R-hat ~ 0 + infinite ESS = too short to assess, not "converged".
    txt = _run_verdict(diag, 0.0, float("inf"), [True], [0.5], float("nan"))
    assert "INCONCLUSIVE" in txt


def test_verdict_inconclusive_no_rhat(diag):
    txt = _run_verdict(diag, None, None, [True], [0.5], 20.0)
    assert "INCONCLUSIVE" in txt


# ── R-hat sign folds ────────────────────────────────────────────────────────────

PARAMS3 = ["M_1", "M_2", "mu"]


def _two_lobe_chain(nsteps=300, nwalkers=4):
    '''Half the walkers stuck in the M_1 < 0 lobe, half in M_1 > 0, all at mu > 0:
    the magnitudes agree perfectly, only the (physical) sign labels disagree.'''
    rng = np.random.default_rng(1)
    ch = np.empty((nsteps, nwalkers, 3))
    lobe = np.where(np.arange(nwalkers) < nwalkers // 2, 1.0, -1.0)
    ch[:, :, 0] = (500 + 5 * rng.standard_normal((nsteps, nwalkers))) * lobe
    ch[:, :, 1] = 800 + 5 * rng.standard_normal((nsteps, nwalkers))
    ch[:, :, 2] = 300 + 5 * rng.standard_normal((nsteps, nwalkers))   # mu > 0
    return ch


def test_magnitude_fold_collapses_sign_lobes(diag):
    ch = _two_lobe_chain()
    diag._REPORT.clear()
    signed = diag.rhat_ess([ch], PARAMS3, 0, 1, "signed", canon=True)
    mag = diag.rhat_ess([ch], PARAMS3, 0, 1, "magnitude", mag=True)
    # Signed R-hat here is *entirely* the unmixed sign lobes; abs() removes exactly
    # that, so the magnitude line is the floor the signed one can fall to at best.
    assert signed["worst_rhat"] > 1.5
    assert mag["worst_rhat"] < 1.1


def test_folds_do_not_mutate_caller_chains(diag):
    ch = _two_lobe_chain()
    before = ch.copy()
    diag._REPORT.clear()
    diag.rhat_ess([ch], PARAMS3, 0, 1, "signed", canon=True)
    diag.rhat_ess([ch], PARAMS3, 0, 1, "magnitude", mag=True)
    # Both folds must work on copies: _emit_mode_occupancy re-reads these raw signs
    # afterwards, and abs() would silently zero out its sign-octant table.
    assert np.array_equal(ch, before)


def test_magnitude_fold_emits_no_verdict(diag):
    # The magnitude line is diagnostic only, so it must never print a verdict word
    # that the reader (or a log grep) could mistake for a convergence certificate.
    diag._REPORT.clear()
    diag.rhat_ess([_two_lobe_chain()], PARAMS3, 0, 1,
                  "|M_1|,|M_2| magnitude fold -- DIAGNOSTIC ONLY", mag=True)
    txt = "\n".join(diag._REPORT)
    assert "DIAGNOSTIC ONLY" in txt
    assert "NOT CONVERGED" not in txt and "converged" not in txt


def test_unestimable_rhat_is_reported_as_such_not_as_zero(diag):
    '''A chain far too short for arviz (the CI smoke run: 18 walkers x 3 post-burn-in
    draws) yields nan for every R-hat and ESS. Those must propagate as nan.

    Regression: the aggregation folded with max()/min() as it went, and max(0.0, nan)
    is 0.0 because `nan > 0.0` is False -- so the summary line claimed "worst
    R-hat = 0.000 (converged)" with "min ESS_bulk = inf", and the criteria table scored
    R-hat as a PASS (its isfinite guard never saw a NaN) while ESS correctly said
    UNKNOWN. Seen in a real CI log for genModels_mcmc.'''
    rng = np.random.default_rng(3)
    tiny = rng.standard_normal((3, 18, 3))          # 3 draws, 18 walkers-as-chains
    diag._REPORT.clear()
    res = diag.rhat_ess([tiny], PARAMS3, 0, 1, "too short")
    txt = "\n".join(diag._REPORT)

    assert not np.isfinite(res["worst_rhat"]), "unestimable R-hat must stay nan"
    assert not np.isfinite(res["min_ess_bulk"]), "unestimable ESS must not become inf"
    assert "0.000 (converged)" not in txt, "must not claim convergence on nan input"
    assert "not estimable" in txt
    # per-param rows must not be flagged ok either
    assert " ok" not in txt.split("worst R-hat")[0], "nan rows must not read as ok"


def test_a_healthy_chain_still_reports_numbers(diag):
    '''Guard the fix from the other side: nan-safe aggregation must not turn a
    perfectly estimable chain into "not estimable".'''
    rng = np.random.default_rng(4)
    ch = rng.standard_normal((400, 6, 3)) * 5 + 100
    diag._REPORT.clear()
    res = diag.rhat_ess([ch], PARAMS3, 0, 1, "healthy")
    assert np.isfinite(res["worst_rhat"]) and res["worst_rhat"] < 1.1
    assert np.isfinite(res["min_ess_bulk"]) and res["min_ess_bulk"] > 0
    assert "not estimable" not in "\n".join(diag._REPORT)


# ── HDF5 -> ntuple layout + collision guard ─────────────────────────────────────

def _make_backend(path, nwalkers=6, ndim=2, nsteps=8):
    '''A tiny real emcee HDFBackend (Gaussian toy + one observable blob).'''
    import emcee

    def lp(theta):
        return -0.5 * float(np.dot(theta, theta)), float(theta.sum())

    backend = emcee.backends.HDFBackend(path)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lp, backend=backend,
                                    blobs_dtype=[("obs0", float)])
    p0 = np.random.default_rng(1).normal(scale=0.5, size=(nwalkers, ndim))
    sampler.run_mcmc(p0, nsteps)
    return backend


def _meta():
    return {
        "in_branches": [
            {"kind": "free", "branch": "IN_p0", "index": 0},
            {"kind": "free", "branch": "IN_p1", "index": 1},
        ],
        "record_branches": [{"name": "obs0", "branch": "SP_obs0"}],
    }


def test_build_ntuple_tree_layout(tmp_path):
    path = str(tmp_path / "chain.h5")
    _make_backend(path)
    tree, n = en.build_ntuple_tree(path, _meta(), discard=0, thin=1)
    assert set(tree) == {"IN_p0", "IN_p1", "SP_obs0", "emcee_logp", "chi2"}
    assert all(len(v) == n for v in tree.values())
    assert np.allclose(tree["chi2"], -2.0 * tree["emcee_logp"])


def test_h5_to_ntuple_branches(tmp_path):
    import uproot
    path = str(tmp_path / "chain.h5")
    _make_backend(path)
    out = str(tmp_path / "ntuple.root")
    n, nbr = en.h5_to_ntuple(path, out, discard=0, thin=1, meta=_meta())
    branches = set(uproot.open(out)["susy"].keys())
    assert {"IN_p0", "IN_p1", "SP_obs0", "emcee_logp", "chi2"} <= branches
    assert nbr == len(branches)
    assert n > 0


def test_ntuple_reserved_branch_collision(tmp_path):
    path = str(tmp_path / "chain.h5")
    _make_backend(path)
    bad = _meta()
    bad["record_branches"] = [{"name": "obs0", "branch": "chi2"}]   # reserved name
    with pytest.raises(ValueError):
        en.build_ntuple_tree(path, bad, discard=0, thin=1)


def test_ntuple_duplicate_branch_collision(tmp_path):
    path = str(tmp_path / "chain.h5")
    _make_backend(path)
    bad = _meta()
    bad["in_branches"][1]["branch"] = "IN_p0"   # two branches claim IN_p0
    with pytest.raises(ValueError):
        en.build_ntuple_tree(path, bad, discard=0, thin=1)


# ── config-schema validation (ModelGenerator._validate_observable_sources) ───────

def _validate(prior="mcmc", steps=None, constraints=None, record=None):
    stub = SimpleNamespace(
        prior=prior,
        steps=steps if steps is not None else [{"name": "SPheno"}],
        constraints=constraints if constraints is not None else [],
        mcmc={"record_observables": record or []},
    )
    ModelGenerator._validate_observable_sources(stub)


def test_validate_ok():
    _validate(
        constraints=[{"name": "h", "source": {"step": "SPheno"}, "uncertainty": 1.0}],
        record=[{"name": "m_h", "branch": "SP_m_h", "source": {"step": "SPheno"}}],
    )  # no raise


def test_validate_missing_source_step():
    with pytest.raises(ValueError):
        _validate(constraints=[{"name": "h", "source": {"step": "NoSuchStep"},
                                "uncertainty": 1.0}])


def test_validate_nonpositive_uncertainty():
    with pytest.raises(ValueError):
        _validate(constraints=[{"name": "h", "source": {"step": "SPheno"},
                                "uncertainty": 0}])


def test_validate_reserved_record_branch():
    with pytest.raises(ValueError):
        _validate(record=[{"name": "x", "branch": "chi2", "source": {"step": "SPheno"}}])


def test_validate_duplicate_record_branch():
    with pytest.raises(ValueError):
        _validate(record=[
            {"name": "x", "branch": "SP_dup", "source": {"step": "SPheno"}},
            {"name": "y", "branch": "SP_dup", "source": {"step": "SPheno"}},
        ])


def test_validate_noop_for_non_mcmc_prior():
    # A non-mcmc prior must skip the whole check, even with an otherwise-bad source.
    _validate(prior="flat",
              constraints=[{"name": "h", "source": {"step": "NoSuchStep"},
                            "uncertainty": -1}])
