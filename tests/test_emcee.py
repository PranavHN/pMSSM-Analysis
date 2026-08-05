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


# ── rank statistic / computational faithfulness ─────────────────────────────────

def _ens(nsteps=3000, nwalkers=8, ndim=3, shift=0.0, scale=1.0, seed=0):
    '''One ensemble of iid draws; `shift`/`scale` perturb the first parameter only.'''
    rng = np.random.default_rng(seed)
    ch = rng.standard_normal((nsteps, nwalkers, ndim))
    ch[:, :, 0] = ch[:, :, 0] * scale + shift
    return ch


def _ar1_ens(nsteps=4000, nwalkers=8, ndim=3, phi=0.98, seed=0):
    '''Strongly autocorrelated ensemble, tau_int = (1+phi)/(1-phi) ~ 99 at phi=0.98.
    Every walker of every ensemble is drawn from the SAME stationary distribution.'''
    rng = np.random.default_rng(seed)
    ch = np.empty((nsteps, nwalkers, ndim))
    ch[0] = rng.standard_normal((nwalkers, ndim))
    s = np.sqrt(1 - phi ** 2)
    for i in range(1, nsteps):
        ch[i] = phi * ch[i - 1] + s * rng.standard_normal((nwalkers, ndim))
    return ch


def _rank(diag, ens, max_tau=5.0, **kw):
    diag._REPORT.clear()
    return diag.rank_uniformity(ens, PARAMS3, 0, max_tau, "t", canon=False, **kw)


def test_identical_ensembles_are_uniform(diag):
    res = _rank(diag, [_ens(seed=s) for s in range(4)])
    assert res["worst_p"] > diag.RANK_P_MIN, "chains from one distribution must pass"
    assert all(s == "uniform" for sh in res["shapes"].values() for s in sh)
    assert "uniform (consistent with one distribution)" in "\n".join(diag._REPORT)


def test_autocorrelated_but_healthy_chains_do_not_fail(diag):
    '''The regression that killed two earlier designs. These chains are strongly
    autocorrelated (tau ~ 99) but all sample the same distribution, so the check MUST
    pass. A chi-square over all 32k draws assuming independence rejects ~always
    (statistic inflated ~17x); scaling it by ESS/N over-corrects by an order of
    magnitude and then never rejects anything. Only thinning by tau is calibrated.'''
    res = _rank(diag, [_ar1_ens(seed=s) for s in range(4)], max_tau=99.0)
    assert res["worst_p"] > diag.RANK_P_MIN, (
        f"autocorrelated-but-healthy chains flagged non-uniform (p={res['worst_p']:.2e}) "
        f"-- the test is not accounting for correlation")


def test_a_shifted_ensemble_is_caught_and_reads_as_a_location_shift(diag):
    '''A location shift must be caught, and the LOCATION contrast must be the dominant
    signature. Asserting the bare label "skewed-high" would be too strict: at a few
    thousand draws a 0.6 sigma shift also moves the quadratic contrast past 3 sigma (a
    shifted normal ranked against the pooled mixture is not a purely linear distortion),
    so the honest label is "mixed" -- with location dominating, which is the real claim.'''
    res = _rank(diag, [_ens(seed=0, shift=0.6)] + [_ens(seed=s) for s in (1, 2, 3)])
    assert res["worst_p"] < diag.RANK_P_MIN
    assert res["worst_param"] == "M_1", "the perturbed parameter must be the worst"
    label, z_loc, z_dis = diag._rank_shape(res["thinned_hist"]["M_1"][0])
    assert label in ("skewed-high", "mixed"), label
    assert z_loc > 0, "the shifted ensemble must sit HIGH in the pooled ranks"
    assert abs(z_loc) > abs(z_dis), "location must dominate for a pure location shift"
    assert "non-uniform -- chains disagree" in "\n".join(diag._REPORT)


def test_an_over_dispersed_ensemble_is_called_u_shaped(diag):
    '''Chains here are MCMC replicates, not SBC refits, so a U-shape means this chain
    is over-dispersed RELATIVE TO THE POOLED sample -- not that a posterior is narrow.'''
    res = _rank(diag, [_ens(seed=0, scale=1.8)] + [_ens(seed=s) for s in (1, 2, 3)])
    assert res["worst_p"] < diag.RANK_P_MIN
    assert res["shapes"]["M_1"][0] == "U-shaped"
    assert "inverted-U" in res["shapes"]["M_1"][1:], \
        "the narrower chains must read as under-dispersed against the pooled sample"


def test_unperturbed_parameters_stay_uniform(diag):
    '''A localised problem must not smear across parameters.'''
    res = _rank(diag, [_ens(seed=0, shift=0.6)] + [_ens(seed=s) for s in (1, 2, 3)])
    for name in ("M_2", "mu"):
        assert res["per_param"][name]["p_holm"] > diag.RANK_P_MIN
        assert all(s == "uniform" for s in res["shapes"][name])


def test_too_few_independent_draws_is_not_estimable(diag):
    '''Never a spurious pass: the same fail-closed rule as the R-hat nan guard.'''
    res = _rank(diag, [_ens(nsteps=40, nwalkers=2, seed=s) for s in range(4)],
                max_tau=500.0)
    assert not np.isfinite(res["worst_p"])
    assert res["n_thinned"] == 0
    txt = "\n".join(diag._REPORT)
    assert "not estimable" in txt
    assert "non-uniform" not in txt and "consistent with one distribution" not in txt


def test_a_non_finite_draw_degrades_only_its_own_parameter(diag):
    '''Ranking nan/inf would silently drop it into an arbitrary bin. Report that
    parameter as unestimable and keep testing the others.'''
    ens = [_ens(nsteps=2000, nwalkers=6, seed=s) for s in range(4)]
    ens[0][5, 2, 1] = np.inf                      # poison one draw of M_2
    res = _rank(diag, ens)
    assert not np.isfinite(res["per_param"]["M_2"]["p_holm"])
    assert res["shapes"]["M_2"] == ["n/a"] * 4
    for name in ("M_1", "mu"):
        assert np.isfinite(res["per_param"][name]["p_holm"]), \
            f"{name} must still be tested when another parameter is poisoned"


def test_a_single_ensemble_falls_back_to_time_segments(diag):
    '''What every chain job's auto-run hits: one backend, so no independent replicates.
    Splitting it in time makes this a stationarity check, and it must say so.'''
    res = _rank(diag, [_ens(nsteps=6000, seed=0)])
    assert res["segmented"] is True
    assert res["n_chains"] == diag.RANK_SEGMENTS
    assert res["worst_p"] > diag.RANK_P_MIN, "a stationary chain must not flag drift"


def test_time_segments_detect_drift(diag):
    '''A single ensemble that is still moving must be caught by the segment split.'''
    ch = _ens(nsteps=6000, seed=0)
    ch[:, :, 0] += np.linspace(0.0, 2.0, ch.shape[0])[:, None]   # steady drift
    res = _rank(diag, [ch])
    assert res["worst_p"] < diag.RANK_P_MIN
    assert res["worst_param"] == "M_1"


def test_holm_adjustment_is_monotone_and_bounded(diag):
    p = [1e-6, 0.02, 0.3, 0.9, float("nan")]
    adj = diag._holm(p)
    finite = adj[:4]
    assert np.all(np.diff(finite) >= 0), "step-down adjusted p must be non-decreasing"
    assert np.all(finite <= 1.0) and np.all(finite >= np.asarray(p[:4]))
    assert not np.isfinite(adj[4]), "nan in must stay nan out"


def test_rank_shape_classifier_labels(diag):
    B = diag.RANK_BINS
    n = 4000
    flat = np.full(B, n / B)
    assert diag._rank_shape(flat)[0] == "uniform"
    rising = np.linspace(0.5, 1.5, B); rising = rising / rising.sum() * n
    assert diag._rank_shape(rising)[0] == "skewed-high"
    falling = rising[::-1]
    assert diag._rank_shape(falling)[0] == "skewed-low"
    bowl = np.abs(np.linspace(-1, 1, B)) + 0.2; bowl = bowl / bowl.sum() * n
    assert diag._rank_shape(bowl)[0] == "U-shaped"
    assert diag._rank_shape(bowl.max() - bowl + 1e-9)[0] == "inverted-U"
    assert diag._rank_shape(np.zeros(B))[0] == "n/a"


def test_folded_blocks_does_not_mutate_caller_chains(diag):
    '''Extends the guarantee rhat_ess already had, now that the fold is shared: mode
    occupancy re-reads the raw signs after these calls, and abs() is not reversible.'''
    ch = _two_lobe_chain()
    before = ch.copy()
    diag._folded_blocks([ch], PARAMS3, 0, 1, canon=True)
    diag._folded_blocks([ch], PARAMS3, 0, 1, mag=True)
    assert np.array_equal(ch, before)


# ── criteria table ──────────────────────────────────────────────────────────────

def _table(diag, rhat=1.01, ess=900, n50=(True,), acc=(0.3,), rank_res=None):
    diag._REPORT.clear()
    diag._emit_criteria_table({"worst_rhat": rhat, "min_ess_bulk": ess,
                               "min_ess_tail": ess, "label": "L"},
                              list(n50), [10000], [10.0], list(acc), 100, 10,
                              rank_res=rank_res)
    return "\n".join(diag._REPORT)


def test_table_summary_block_reports_every_check(diag):
    txt = _table(diag)
    assert "SUMMARY" in txt and "hard 3/3 PASS" in txt
    for name in ("chain length vs tau", "Gelman-Rubin R-hat", "bulk effective sample size",
                 "mean acceptance fraction", "tail effective sample size", "burn-in adequacy"):
        assert f"] {name}" in txt, f"{name} missing from the summary block"


def test_table_gains_a_rank_row_when_the_check_ran(diag):
    rank_res = {"worst_p": 1e-9, "worst_param": "meL", "n_chains": 4, "segmented": False,
                "shapes": {"meL": ["U-shaped", "uniform", "uniform", "uniform"]}}
    txt = _table(diag, rank_res=rank_res)
    assert "7  rank uniformity" in txt
    assert "4 advisory" in txt
    assert "[non-uniform] rank uniformity" in txt
    assert "meL" in txt and "U-shaped" in txt
    assert "NOT simulation-based calibration" in txt, \
        "the table must not let a reader mistake this for calibration evidence"


def test_a_failing_rank_check_does_not_change_the_verdict(diag):
    '''It is advisory: R-hat already gates inter-chain agreement.'''
    rank_res = {"worst_p": 1e-30, "worst_param": "meL", "n_chains": 4,
                "segmented": False, "shapes": {"meL": ["U-shaped"]}}
    assert "3 hard" in _table(diag, rank_res=rank_res)
    out = _run_verdict(diag, 1.01, 900, [True], [0.3], 10.0)
    assert "CONVERGED" in out and "NOT CONVERGED" not in out


def test_hard_verdicts_is_the_shared_source_of_truth(diag):
    '''The table and the verdict must not be able to disagree. Both read _hard_verdicts,
    so one changed threshold cannot update only one of them.'''
    rows = diag._hard_verdicts({"worst_rhat": 2.0, "min_ess_bulk": 10}, [False])
    assert [ok for _n, ok, _m in rows] == [False, False, False]
    table = _table(diag, rhat=2.0, ess=10, n50=(False,))
    verdict = _run_verdict(diag, 2.0, 10, [False], [0.3], 10.0)
    assert "hard 0/3 PASS" in table
    for _name, _ok, msg in rows:
        assert msg in verdict, f"verdict lost the failure message {msg!r}"


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
