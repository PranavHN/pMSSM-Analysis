#!/usr/bin/env python
'''MCMC diagnostics for emcee HDF5 backends (Run3ModelGen pMSSM scan).

Reads one or more `emcee.backends.HDFBackend` files (plus their sidecar
`<backend>.meta.json`) and reports the standard emcee/arviz diagnostics:

  * mean + per-walker acceptance fraction
  * integrated autocorrelation time tau per parameter (get_autocorr_time)
  * tau-vs-N convergence curve (the emcee autocorrelation tutorial plot)
  * trace plots and log-probability trace
  * corner plot of the marginal/joint posteriors
  * scatter + marginal histograms of each free input dim vs the constrained
    "target" observable (auto-detected from the config, e.g. m_h or Omega),
    coloured by log-prob -- one figure per input dim (--target to override)
  * with more than one ensemble, the corner and input-vs-target figures are ALSO
    written for every ensemble pooled (tag "all"), matching the cross-ensemble
    R-hat/ESS: the stored chain is already a correctly weighted sample (one row =
    one unit of weight), so concatenating ensembles simply adds weight
  * Gelman-Rubin R-hat and bulk/tail ESS via arviz, both WITHIN each ensemble
    (walkers as chains) and ACROSS the independent ensembles (the strong test),
    plus a non-gating |M_1|,|M_2| magnitude-fold line as a reference floor for
    how much of the signed R-hat is unmixed sign lobes
  * a rank-statistic uniformity check on ALL samples pooled (rank_all.png,
    rank_ecdf_all.png): ranks pooled over every ensemble and walker, histogrammed
    per chain, which must be uniform if every chain targets the same distribution.
    Reports a chi-square p-value and the histogram SHAPE per chain, since the shape
    is the diagnostic content. This is the Vehtari et al. rank plot, NOT
    simulation-based calibration -- it certifies that the chains agree with each
    other, never that the posterior is calibrated
  * a criteria table: every convergence requirement, why it exists (with a
    literature reference), and whether this scan meets it, with a scannable
    pass/fail summary -- followed by the CONVERGED / NOT CONVERGED verdict, which
    is gated on the "hard" rows

It can also bridge each backend to a ROOT "susy" ntuple (`--to-ntuple <dir>`)
so the existing `extractModels` (and other ntuple consumers) keep working.

Self-contained: the diagnostics + plots import only numpy, scipy, emcee,
matplotlib, corner, arviz, uproot and the standard library (all in the default
pixi environment); PyYAML is used opportunistically to auto-detect the target
observable (falls back if absent). Only `--to-ntuple` additionally imports the
(physics-free) `Run3ModelGen.emcee_ntuple` writer.

Usage:
    pixi run emcee_diagnostics.py --backend chain.h5 [chain2.h5 ...] \
        --outdir diag [--discard auto --thin auto] [--to-ntuple ntuples/]
    pixi run emcee_diagnostics.py --glob '/path/ens_*/chain.h5' --outdir diag
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: must precede pyplot import

import argparse
import glob as globmod
import json
import os
from pathlib import Path

import numpy as np
import emcee
import matplotlib.pyplot as plt

import logging
import structlog
log = structlog.get_logger("emcee_diagnostics")
structlog.stdlib.recreate_defaults(log_level=logging.INFO)  # timestamps + levels + logger name

# recreate_defaults() puts the ROOT logger at INFO with a bare "%(message)s" format,
# so third-party INFO records print as unformatted lines among ours. arviz emits three
# on import ("arviz_<base|stats|plots> not installed"): the optional ArviZ-1.0 rewrite
# packages, reachable only via arviz.preview and unused here (rhat/ess come from the
# classic arviz.stats), so there is nothing to install and nothing to report. Must be
# set before the first arviz import, which `import corner` triggers via corner.arviz_corner.
logging.getLogger("arviz").setLevel(logging.WARNING)

# Diagnostics output is both LOGGED (structlog) and captured line-by-line into a
# plain-text report written to <outdir>/diagnostics.txt at the end of main().
_REPORT: list = []


def _emit(msg: str, warn: bool = False) -> None:
    '''Log a diagnostics line AND capture it for the <outdir>/diagnostics.txt report.'''
    _REPORT.append(msg)
    (log.warning if warn else log.info)(msg)


# ── convergence criteria ────────────────────────────────────────────────────
# One definition of each threshold, used by BOTH the criteria table and the final
# verdict so the two can never drift apart. Provenance is in _CRITERIA_REFS.
TAU_FACTOR = 50          # required chain length in units of tau        [1,2]
RHAT_MAX = 1.05          # worst tolerated Gelman-Rubin R-hat           [3,4]
ESS_MIN = 400            # required bulk (and desired tail) ESS         [4]
ACC_BAND = (0.2, 0.5)    # healthy mean acceptance fraction             [2,5]
RANK_BINS = 10           # rank-histogram bins for the uniformity test  [4,6]
RANK_P_MIN = 0.01        # Holm-corrected p below this = non-uniform    [4,6]
RANK_SEGMENTS = 4        # single-ensemble fallback: time segments      [4]
RANK_Z_FLAT = 3.0        # |z| below this = no shape called             [6]

_CRITERIA_REFS = [
    "[1] Goodman & Weare 2010, Comm. App. Math. Comp. Sci. 5, 65 -- the affine-invariant",
    "    ensemble sampler (stretch move) that emcee implements.",
    "[2] Foreman-Mackey, Hogg, Lang & Goodman 2013, PASP 125, 306 -- arXiv:1202.3665 --",
    "    emcee itself: tau estimation, the N > 50*tau rule of thumb, the acceptance band,",
    "    and the discard ~ 2-3*tau / thin ~ tau/2 post-processing convention.",
    "[3] Gelman & Rubin 1992, Statist. Sci. 7, 457 -- the original between-chain /",
    "    within-chain variance ratio R-hat.",
    "[4] Vehtari, Gelman, Simpson, Carpenter & Buerkner 2021, Bayesian Analysis 16, 667 --",
    "    arXiv:1903.08008 -- rank-normalized split-R-hat and bulk/tail ESS exactly as arviz",
    "    computes them here, with the ESS >= 400 and R-hat < 1.01 recommendations.",
    "[5] Roberts, Gelman & Gilks 1997, Ann. Appl. Probab. 7, 110 -- the optimal-scaling",
    "    result (acceptance -> 0.234 in high dimension) behind the healthy band.",
    "[6] Talts, Betancourt, Simpson, Vehtari & Gelman 2018 -- arXiv:1804.06788 --",
    "    simulation-based calibration: the rank statistic, and reading pathology off the",
    "    shape of its histogram (U / inverted-U / skewed). SBC itself needs many refits",
    "    against simulated ground truth and is NOT what is computed here; see the rank",
    "    uniformity block for exactly what this check does and does not certify.",
    "[7] Saeilynoja, Buerkner & Vehtari 2022, Stat. Comput. 32, 32 -- arXiv:2103.10522 --",
    "    the uniform-ECDF difference plot with a simultaneous confidence band, used for",
    "    the rank_ecdf figure.",
]


def _hard_verdicts(rhat_res, n50_flags):
    '''The three gating criteria, as ``[(name, ok, failure_message)]``.

    THE single source of truth: both the criteria table and the final verdict read this,
    so a threshold or a criterion can never be changed in one and forgotten in the
    other. Previously each re-derived the same three conditions from the module
    constants independently, which is exactly the kind of duplication that drifts.

    `ok` is tri-state: True / False / None = not estimable.'''
    worst_rhat = rhat_res["worst_rhat"] if rhat_res else float("nan")
    min_ess = rhat_res["min_ess_bulk"] if rhat_res else float("nan")
    all_n50 = bool(n50_flags) and all(n50_flags)
    return [
        ("chain length vs tau", all_n50 if n50_flags else None,
         f"chain too short: N < {TAU_FACTOR}*tau for at least one ensemble"),
        ("Gelman-Rubin R-hat",
         (worst_rhat < RHAT_MAX) if np.isfinite(worst_rhat) else None,
         f"R-hat {worst_rhat:.3f} >= {RHAT_MAX}"),
        ("bulk effective sample size",
         (min_ess >= ESS_MIN) if np.isfinite(min_ess) else None,
         f"ESS_bulk {min_ess:.0f} < {ESS_MIN}"),
    ]


def _emit_criteria_table(rhat_res, n50_flags, n_iters, tau_maxes,
                         acc_means, discard, plateau, rank_res=None) -> None:
    '''Tabulate every criterion the scan is judged on: what it requires, why it
    exists (with a literature reference), and whether THIS scan meets it.

    Printed just before the verdict and captured into diagnostics.txt with the rest
    of the report. The hard rows are exactly what _emit_convergence_verdict gates
    CONVERGED on; the advisory rows are reported but never veto it.'''

    def _fmt(v, spec=".3f"):
        return "n/a" if v is None or not np.isfinite(v) else format(v, spec)

    worst_rhat = rhat_res["worst_rhat"] if rhat_res else float("nan")
    min_bulk = rhat_res["min_ess_bulk"] if rhat_res else float("nan")
    min_tail = rhat_res.get("min_ess_tail", float("nan")) if rhat_res else float("nan")
    scope = rhat_res.get("label", "?") if rhat_res else "not computed (arviz step skipped)"
    acc = float(np.nanmean(acc_means)) if acc_means else float("nan")

    # Chain length: quote the ensemble that sits furthest from the N > TAU_FACTOR*tau line.
    ratios = [n / (TAU_FACTOR * t) if (np.isfinite(t) and t > 0) else float("nan")
              for n, t in zip(n_iters, tau_maxes)]
    if ratios and np.any(np.isfinite(ratios)):
        k = int(np.nanargmin(ratios))
        len_val = f"N={n_iters[k]} vs {TAU_FACTOR}*tau={TAU_FACTOR * tau_maxes[k]:.0f}"
    else:
        len_val = "n/a"

    # ok flags come from _hard_verdicts so the table and the verdict cannot disagree
    hard_ok = {name: ok for name, ok, _ in _hard_verdicts(rhat_res, n50_flags)}

    # (name, requirement, measured, ok, word-if-not-ok, why-lines)
    hard = [
        ("chain length vs tau", f"N > {TAU_FACTOR}*tau, every ens", len_val,
         hard_ok["chain length vs tau"], "FAIL",
         ["tau is the number of steps between effectively independent samples. Below",
          f"~{TAU_FACTOR}*tau the tau estimate is only a lower bound, so the ESS and R-hat",
          "derived from it cannot be trusted either.  [1,2]"]),
        ("Gelman-Rubin R-hat", f"worst R-hat < {RHAT_MAX}", _fmt(worst_rhat),
         hard_ok["Gelman-Rubin R-hat"], "FAIL",
         ["Between-chain over within-chain variance: > 1 means the chains still disagree",
          "about the posterior (stuck walkers, an unmixed mode). arviz computes the",
          "rank-normalized split form; < 1.01 is the stricter modern target.  [3,4]"]),
        ("bulk effective sample size", f"min ESS_bulk >= {ESS_MIN}", _fmt(min_bulk, ".0f"),
         hard_ok["bulk effective sample size"], "FAIL",
         ["Independent-sample count behind the bulk of the posterior (mean, median). The",
          f"recommended floor is {ESS_MIN}: below it the Monte-Carlo error on posterior",
          "summaries -- and the R-hat/ESS estimates themselves -- stop being reliable.  [4]"]),
    ]
    advisory = [
        ("mean acceptance fraction", f"{ACC_BAND[0]} <= acc <= {ACC_BAND[1]}", _fmt(acc),
         (ACC_BAND[0] <= acc <= ACC_BAND[1]) if np.isfinite(acc) else None, "off band",
         ["Near 0 the walkers never move; near 1 the proposals are too timid. Both inflate",
          "tau. In 1-2 free dimensions the stretch move accepts intrinsically often (its",
          "z^(ndim-1) factor is ~1), so a high value there is expected rather than a fault;",
          "raise StretchMove's `a` to bring it down.  [2,5]"]),
        ("tail effective sample size", f"min ESS_tail >= {ESS_MIN}", _fmt(min_tail, ".0f"),
         (min_tail >= ESS_MIN) if np.isfinite(min_tail) else None, "low",
         ["Same idea for the distribution's tails (5%/95% quantiles), which converge more",
          "slowly than the bulk. Low tail ESS means credible intervals are noisy even when",
          "the mean looks well determined.  [4]"]),
        ("burn-in adequacy", "discard >= logp plateau", f"{discard} >= {plateau}",
         discard >= plateau, "short",
         ["The discard must cover the initial transient, estimated independently here from",
          "where the log-probability trace plateaus. emcee's convention is 2-3*tau, which is",
          f"what the sampler applied ({discard} iterations).  [2]"]),
    ]

    if rank_res is not None:
        rp = rank_res.get("worst_p", float("nan"))
        rw = rank_res.get("worst_param")
        shapes = sorted({s for sh in rank_res.get("shapes", {}).values() for s in sh
                         if s not in ("uniform", "n/a")})
        scope_txt = ("time segments of one ensemble" if rank_res.get("segmented")
                     else f"{rank_res.get('n_chains', 0)} ensembles")
        detail = (f"worst {rw}" if rw else "n/a")
        if shapes:
            detail += "; shapes seen: " + ", ".join(shapes)
        advisory.append(
            ("rank uniformity", f"Holm p >= {RANK_P_MIN}",
             "n/a" if not np.isfinite(rp) else f"{rp:.1e}",
             (rp >= RANK_P_MIN) if np.isfinite(rp) else None, "non-uniform",
             ["Ranks pooled over all samples, histogrammed per chain, must be uniform if",
              "every chain samples the same distribution; the SHAPE says how they differ",
              "(U = over-dispersed vs pooled, inverted-U = under-dispersed/trapped,",
              "skewed = shifted). Chains here are " + scope_txt + "; " + detail + ".",
              "NOT simulation-based calibration -- it certifies the chains agree with each",
              "other, not that the posterior is calibrated.  [4,6,7]"]))

    words = {True: "PASS", False: "FAIL", None: "UNKNOWN"}
    soft = {True: "ok", False: None, None: "n/a"}     # False -> the row's own bad word
    bar = "=" * 88

    # Compact, scannable summary first: the detailed rows below explain WHY each
    # criterion exists, but the question a reader arrives with is "what failed?".
    n_hard_pass = sum(1 for r in hard if r[3] is True)
    n_soft_ok = sum(1 for r in advisory if r[3] is True)
    summary = [f"SUMMARY  hard {n_hard_pass}/{len(hard)} PASS   "
               f"advisory {n_soft_ok}/{len(advisory)} ok"]
    for group, rows in (("hard", hard), ("advisory", advisory)):
        cells = []
        for name, _req, _val, ok, bad, _why in rows:
            word = words[ok] if group == "hard" else (soft[ok] or bad)
            cells.append(f"[{word}] {name}")
        # wrap at three per line so the block stays readable in a terminal
        for i in range(0, len(cells), 3):
            summary.append("    " + "   ".join(cells[i:i + 3]))

    out = [bar,
           f"CONVERGENCE CRITERIA  ({len(hard)} hard -- all must pass for CONVERGED; "
           f"{len(advisory)} advisory)",
           f"  R-hat / ESS measured on: {scope}",
           ""]
    out.extend(summary)
    out.append("")
    n = 0
    for group, rows in (("hard", hard), ("advisory -- reported, never vetoes the verdict",
                                         advisory)):
        out.append(f"  --- {group} ---")
        for name, req, val, ok, bad, why in rows:
            n += 1
            verdict = words[ok] if group == "hard" else (soft[ok] or bad)
            out.append(f"  {n}  {name:<26s} {req:<24s} {val:<22s} {verdict}")
            for line in why:
                out.append(f"        {line}")
        out.append("")
    out.append("REFERENCES")
    out.extend(f"  {r}" for r in _CRITERIA_REFS)
    out.append(bar)
    _emit("\n".join(out))


def _emit_convergence_verdict(rhat_res, n50_flags, acc_means, max_tau) -> None:
    '''Final statement: are the diagnostics OK / has the scan converged?

    Converged <=> every ensemble has N > 50*tau AND worst R-hat < 1.05 AND
    min ESS_bulk >= 400. Mean acceptance (healthy ~0.2-0.5) is advisory: off-band
    acceptance flags slow mixing but does not by itself veto a chain that already
    meets the R-hat / ESS / autocorrelation criteria.'''
    worst_rhat = rhat_res["worst_rhat"] if rhat_res else float("nan")
    min_ess = rhat_res["min_ess_bulk"] if rhat_res else float("nan")
    acc = float(np.nanmean(acc_means)) if acc_means else float("nan")

    # A chain too short to even estimate tau / R-hat / ESS is NOT converged — say so
    # rather than letting unestimable values slip past the thresholds below. rhat_ess
    # now propagates nan for "not estimable" (it used to fold NaNs into R-hat 0.0 and
    # ESS inf, which is what the `worst_rhat > 0.5` clause was added to catch); the
    # clause is kept as defence-in-depth against any future degenerate-but-finite value.
    if not (np.isfinite(max_tau) and np.isfinite(worst_rhat) and worst_rhat > 0.5
            and np.isfinite(min_ess)):
        _bar = "=" * 70
        _emit(f"{_bar}\nINCONCLUSIVE - chain too short to assess convergence: tau / R-hat / "
              f"ESS are not reliably estimable (need N >> tau). Run more steps, then "
              f"re-diagnose.\n{_bar}", warn=True)
        return

    # Same rows the criteria table renders -- see _hard_verdicts.
    fails = [msg for _name, ok, msg in _hard_verdicts(rhat_res, n50_flags) if ok is not True]
    acc_note = "" if (ACC_BAND[0] <= acc <= ACC_BAND[1]) else \
        f"  [mean acceptance {acc:.2f} outside the healthy ~{ACC_BAND[0]}-{ACC_BAND[1]} band]"

    bar = "=" * 70
    if not fails:
        _emit(f"{bar}\nCONVERGED - diagnostics OK: N > {TAU_FACTOR}*tau, "
              f"worst R-hat = {worst_rhat:.3f} < {RHAT_MAX}, "
              f"min ESS_bulk = {min_ess:.0f} >= {ESS_MIN}, mean acceptance = {acc:.2f}.{acc_note}\n{bar}")
    else:
        _emit(f"{bar}\nNOT CONVERGED - " + "; ".join(fails) +
              f".\n  (worst R-hat = {worst_rhat:.3f}, min ESS_bulk = {min_ess:.0f}, "
              f"mean acceptance = {acc:.2f}){acc_note}\n"
              f"  Run longer (raise max_steps) and/or add walkers, then re-diagnose.\n{bar}",
              warn=True)


# ── loading ─────────────────────────────────────────────────────────────────

def load_backend(path: str) -> dict:
    '''Open a backend read-only and load its sidecar metadata.'''
    meta_path = path + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    backend = emcee.backends.HDFBackend(path, read_only=True)
    return {"path": path, "backend": backend, "meta": meta,
            "iteration": int(backend.iteration)}


# ── autocorrelation ─────────────────────────────────────────────────────────

def autocorr_time(chain: np.ndarray, tol: int = 50):
    '''Integrated autocorrelation time per parameter for a (nsteps, nwalkers,
    ndim) chain. Returns (tau_array, ok_flag). ok_flag is False when the chain
    is shorter than tol*tau (estimate is then a lower bound).'''
    try:
        tau = emcee.autocorr.integrated_time(chain, tol=tol)
        ok = True
    except emcee.autocorr.AutocorrError as err:
        tau = getattr(err, "tau", None)
        if tau is None:
            tau = emcee.autocorr.integrated_time(chain, tol=0)
        ok = False
    tau = np.asarray(tau)
    # A degenerate / stuck / all-penalty chain yields NaN tau; integrated_time does NOT
    # raise for it (tol*NaN > n is False), so guard here — a non-finite tau is never a
    # reliable "N > tol*tau" pass.
    if not np.all(np.isfinite(tau)):
        ok = False
    return tau, ok


# ── burn-in ─────────────────────────────────────────────────────────────────

def resolve_burnin(max_tau, n_iter, burn_in_floor=0, factor=3.0):
    '''Iterations to discard as burn-in. Delegates to the single source of truth,
    Run3ModelGen.emcee_ntuple.resolve_burnin, so this diagnostic and the sampler's
    ntuple can never drift on the discard. floor/factor come from the backend's
    .meta.json ('burn_in' / 'burn_in_factor').'''
    return _from_emcee_ntuple("resolve_burnin")(max_tau, n_iter, burn_in_floor, factor)


def suggested_burnin_from_logprob(log_prob):
    '''Estimate burn-in from the mean log-prob trace: the first iteration where
    the walker-mean log-prob reaches its equilibrium band (plateau level minus
    one plateau-sigma) -- i.e. the end of the initial climb from the
    over-dispersed start. Returns 0 when no transient is visible.

    log_prob: (nsteps, nwalkers) array (may contain -inf for rejected points).'''
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(np.isfinite(log_prob), log_prob, np.nan), axis=1)
    n = len(m)
    finite = np.isfinite(m)
    if n < 10 or finite.sum() < 5:
        return 0
    tail = m[n // 2:]
    tail = tail[np.isfinite(tail)]
    if tail.size == 0:
        return 0
    level = float(np.median(tail))
    scale = float(np.std(tail)) or 1e-12
    reached = np.where(finite & (m >= level - scale))[0]
    return int(reached[0]) if reached.size else n // 2


def plot_tau_vs_n(chain: np.ndarray, params, outpath: str):
    '''emcee-tutorial convergence plot: tau estimate vs chain length N per parameter
    against the N/50 line, PLUS the same data as tau/N.

    The second panel is the one to read, because the first is easy to misread. While
    N < 50*tau the estimator is in its downward-biased regime: it can only accumulate
    autocorrelation out to a window it can afford, and mean-subtracting a finite series
    forces the estimated ACF negative at large lag (by O(tau/N)), capping the positive
    area it can sum. The result is tau_hat ~ N/const for EVERY coordinate regardless of
    its physics -- which on a log-log tau-vs-N plot looks like unbounded linear growth
    and invites the conclusion that tau diverges and running longer cannot help.

    Plotted as tau/N the artifact is unmistakable: a FLAT line means tau_hat is pinned to
    the run length and is only a lower bound. Real convergence appears as the curve
    peeling downward and crossing 1/50 -- which is exactly the N > 50*tau criterion, so
    the horizontal reference line is the target rather than a diagonal to compare against.
    '''
    nsteps = chain.shape[0]
    ns = np.unique(np.logspace(np.log10(max(10, nsteps // 100)),
                               np.log10(nsteps), 12).astype(int))
    ns = ns[ns > 5]
    taus = np.full((len(ns), chain.shape[2]), np.nan)
    for i, n in enumerate(ns):
        try:
            taus[i] = emcee.autocorr.integrated_time(chain[:n], tol=0)
        except Exception:
            pass
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for p in range(chain.shape[2]):
        ax1.loglog(ns, taus[:, p], "o-", ms=3, label=params[p])
        ax2.loglog(ns, taus[:, p] / ns, "o-", ms=3, label=params[p])
    ax1.loglog(ns, ns / 50.0, "--k", label=r"$\tau = N/50$")
    ax1.set_xlabel("number of samples N")
    ax1.set_ylabel(r"$\tau$ estimate")
    ax1.set_title("Autocorrelation-time convergence")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3, which="both")

    ax2.axhline(1.0 / TAU_FACTOR, color="k", ls="--", lw=1.2,
                label=fr"target $\tau/N = 1/{TAU_FACTOR}$")
    ax2.set_xlabel("number of samples N")
    ax2.set_ylabel(r"$\tau\,/\,N$")
    ax2.set_title(fr"$\tau/N$: flat $\Rightarrow$ pinned to N (lower bound)"
                  "\n" fr"must fall to $1/{TAU_FACTOR}$ to converge")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


# ── trace / logprob / corner ────────────────────────────────────────────────

def plot_traces(chain, params, outpath, max_walkers=40):
    ndim = chain.shape[2]
    fig, axes = plt.subplots(ndim, 1, figsize=(9, 1.6 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    nw = min(max_walkers, chain.shape[1])
    for p in range(ndim):
        axes[p].plot(chain[:, :nw, p], alpha=0.3, lw=0.5)
        axes[p].set_ylabel(params[p], fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Walker traces")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_logprob(backend, outpath, discard=0):
    '''Log-prob history as the ensemble mean + 16-84% walker band (not 256
    overplotted lines), with the burn-in region shaded and the discard marked;
    plus the log-prob distribution pre- vs post-burn-in.'''
    lp = backend.get_log_prob()                       # (nsteps, nwalkers)
    nsteps = lp.shape[0]
    discard = int(max(0, min(discard, nsteps - 1)))
    with np.errstate(invalid="ignore"):
        m = np.where(np.isfinite(lp), lp, np.nan)
    steps = np.arange(nsteps)
    mean = np.nanmean(m, axis=1)
    p16 = np.nanpercentile(m, 16, axis=1)
    p84 = np.nanpercentile(m, 84, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2),
                           gridspec_kw={"width_ratios": [2, 1]})
    # left: mean trace + walker spread, burn-in shaded
    ax[0].fill_between(steps, p16, p84, color="#2c6fbb", alpha=0.25, lw=0,
                       label="walker 16-84%")
    ax[0].plot(steps, mean, color="#2c6fbb", lw=1.2, label="ensemble mean")
    if discard > 0:
        ax[0].axvspan(0, discard, color="#d62728", alpha=0.10)
        ax[0].axvline(discard, color="#d62728", ls="--", lw=1.3,
                      label=f"burn-in discard = {discard}")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(r"log-probability = $-\chi^2/2$")
    ax[0].set_title("log-prob trace (full chain; burn-in shaded)")
    # robust y-range so the deep initial dip doesn't compress the plateau
    fm = mean[np.isfinite(mean)]
    if fm.size:
        ylo = np.percentile(fm, 2)
        yhi = np.nanmax(p84[np.isfinite(p84)])
        pad = 0.05 * (yhi - ylo + 1e-9)
        ax[0].set_ylim(ylo - pad, yhi + pad)
    ax[0].legend(fontsize=8, loc="lower right")
    # right: pre- vs post-burn-in log-prob distributions
    post = m[discard:].ravel(); post = post[np.isfinite(post)]
    pre = (m[:discard].ravel() if discard > 0 else np.array([]))
    pre = pre[np.isfinite(pre)]
    ref = post if post.size else m[np.isfinite(m)]
    bins = (_grid_bins(ref[ref >= np.nanpercentile(ref, 1)], 60)
            if ref.size else 60)
    if pre.size:
        ax[1].hist(pre, bins=bins, density=True, color="#d62728", alpha=0.5,
                   label=f"pre burn-in (<{discard})")
    if post.size:
        ax[1].hist(post, bins=bins, density=True, color="#2c6fbb", alpha=0.6,
                   label=f"post burn-in (>={discard})")
    ax[1].set_xlabel(r"log-probability = $-\chi^2/2$")
    ax[1].set_ylabel("density")
    ax[1].set_title("log-prob: pre vs post burn-in")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)


def plot_corner(flat_chain, params, outpath, target_vals=None, target_name=None):
    import corner
    from matplotlib.lines import Line2D
    ndim = flat_chain.shape[1]
    fig = corner.corner(
        flat_chain, labels=params,
        bins=40, smooth=1.0, smooth1d=1.0,          # smooth jagged 2D/1D histograms
        show_titles=True, title_fmt=".2g",
        title_quantiles=[0.16, 0.5, 0.84],          # titles: median +/- 1sigma
        quantiles=None,                             # suppress corner's 1-colour lines
        title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 13},
        max_n_ticks=4,
        plot_datapoints=False, plot_density=False,  # drop noisy scatter + raw density
        fill_contours=True, levels=(0.393, 0.865, 0.989),  # filled 1,2,3-sigma (2D)
        color="#2c6fbb",
        hist_kwargs={"color": "#2c6fbb", "linewidth": 1.4},
        contour_kwargs={"linewidths": 0.7},
    )
    # Colour-coded quantile lines on each 1-D (diagonal) panel + one figure legend.
    q = np.percentile(flat_chain, [16, 50, 84], axis=0)          # (3, ndim)
    qstyle = [("16th percentile", "#e8a33d", "--"),
              ("median (50th)",   "#d62728", "-"),
              ("84th percentile", "#2ca02c", "--")]
    axes = np.array(fig.axes).reshape(ndim, ndim)
    for i in range(ndim):
        for k, (_, col, ls) in enumerate(qstyle):
            axes[i, i].axvline(q[k, i], color=col, ls=ls, lw=1.3)
    for ax in fig.get_axes():
        ax.tick_params(labelsize=9)
    handles = [Line2D([0], [0], color=col, ls=ls, lw=1.8, label=lab)
               for lab, col, ls in qstyle]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               fontsize=13, frameon=True, title="1-D marginal quantiles",
               title_fontsize=13)
    # Constrained-target (e.g. Omega) marginal, in the empty upper-right triangle.
    tv = np.asarray(target_vals, float) if target_vals is not None else np.array([])
    tv = tv[np.isfinite(tv)]
    if tv.size > 10:
        axin = fig.add_axes([0.62, 0.55, 0.31, 0.24])           # figure coords
        axin.hist(tv, bins=_grid_bins(tv, 50), color="#6b6b6b", alpha=0.85)
        for k, (_, col, ls) in enumerate(qstyle):
            axin.axvline(np.percentile(tv, [16, 50, 84][k]), color=col, ls=ls, lw=1.3)
        axin.set_title(f"{target_name} (constrained target)", fontsize=13)
        axin.set_xlabel(target_name, fontsize=12)
        axin.set_ylabel("count", fontsize=11)
        axin.tick_params(labelsize=9)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── input-parameter vs target-observable scatter/histograms ──────────────────

def infer_target_obs(meta, override=None):
    '''Blob field name of the constrained observable (the scan "target"), used
    for the input-vs-target plots. Matches the config's first POSTERIOR-SHAPING
    constraint to a recorded observable; falls back to the first recorded
    observable. `override` (CLI --target) wins. Returns None if nothing
    recordable is found.

    Shaping means gaussian / upper_bound / lower_bound -- a `hard_window` is a VETO
    (0 inside, inf outside) that carves out the support without expressing anything
    the posterior is fitted to, so it is never the interesting plot axis. This used
    to take the first constraint outright, which meant a config listing a validity
    veto first (`higgs_computed`, m_h > 0) produced every input-vs-target plot
    against m_h instead of the relic density. Skipping vetoes makes the choice
    independent of the order constraints happen to be written in.

    Reading the config needs PyYAML; if unavailable it silently falls back.'''
    rec_names = list(meta.get("record_names", []) or [])
    if override:
        return override
    cfg_path = meta.get("config_file")
    if cfg_path and os.path.exists(cfg_path):
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path))
            cons = cfg.get("constraints") or []
            recobs = (cfg.get("mcmc") or {}).get("record_observables") or []
            shaping = [c for c in cons if c.get("type") != "hard_window"]
            for con in (shaping or cons):    # vetoes only as a last resort
                csrc = con.get("source")
                for ro in recobs:            # match constraint source -> record name
                    if ro.get("source") == csrc and ro.get("name") in rec_names:
                        return ro["name"]
                if con.get("name") in rec_names:
                    return con["name"]
        except Exception:
            pass
    return rec_names[0] if rec_names else None


def _grid_bins(values, target_nbins=50):
    '''Histogram bin edges that MATCH a discrete recording grid, so a coarsely
    quantized observable is shown at its native precision instead of aliasing into
    a moire/comb. The canonical case is micrOMEGAs Omega stored at the tool's own
    %.2e precision, whose grid is MAGNITUDE-DEPENDENT: the step is 1e-3 near 0.12
    but 1e-4 below 0.1 (and 1e-5 below 0.01).

    The grid unit is taken from the DENSE CORE (the sample inter-quartile range),
    NOT the global-minimum spacing. Using the global min (the finest tail step) is
    what produced two artifacts: (1) a bin width that is a NON-integer multiple of
    the bulk step -> some bins swallow 2 grid values, others 3 -> the alternating
    comb; and (2) on a count axis, bins 10x narrower below the 0.1 magnitude
    boundary hold ~10x fewer counts -> a spurious "cliff" at 0.1. A width that is
    an integer number of BULK steps is automatically a whole multiple of every
    finer tail step too (the %.2e grids are nested powers of ten), so it aliases
    nowhere and the counts stay continuous across magnitude boundaries.

    Falls back to `target_nbins` uniform bins when the data is effectively
    continuous (input parameters, log-prob).'''
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return target_nbins
    u = np.unique(v)
    if u.size < 2 or u.size > min(20000, 0.5 * v.size):
        return target_nbins                       # effectively continuous
    # Representative grid step in the bulk (IQR), so the finer tail grid below a
    # magnitude boundary does not set the bin width.
    q1, q3 = np.percentile(v, [25, 75])
    core = u[(u >= q1) & (u <= q3)]
    d = np.diff(core if core.size >= 2 else u)
    d = d[d > 0]
    if d.size == 0:
        return target_nbins
    g = float(np.median(d))                        # bulk grid step
    lo, hi = float(u[0]), float(u[-1])
    if g <= 0 or hi <= lo:
        return target_nbins
    k = max(1, int(round(((hi - lo) / target_nbins) / g)))   # width = k*g ~ range/nbins
    w = k * g
    if (hi - lo) / w > 10000:
        return target_nbins
    start = (np.floor(lo / g) - 0.5) * g           # edges on half-grid points: no level lands on an edge
    return np.arange(start, hi + w, w)


def _joint_plot(x, xlabel, y, ylabel, logp, scatter_idx, outpath, bins=50):
    '''Scatter of x vs y + marginal histograms. The scatter is coloured by logp
    and may be sub-sampled via scatter_idx; the histograms use all points.'''
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.10, right=0.97, bottom=0.09, top=0.94,
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_c = fig.add_subplot(gs[0, 1]); ax_c.axis("off")
    xs, ys = x[scatter_idx], y[scatter_idx]
    if logp is not None:
        sc = ax.scatter(xs, ys, c=logp[scatter_idx], s=7, alpha=0.5, cmap="viridis",
                        edgecolors="none", rasterized=True)
        cax = ax_c.inset_axes([0.30, 0.08, 0.20, 0.86])
        fig.colorbar(sc, cax=cax).set_label(r"emcee $\log p=-\chi^2/2$", fontsize=8)
    else:
        ax.scatter(xs, ys, s=7, alpha=0.4, color="steelblue",
                   edgecolors="none", rasterized=True)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax_x.hist(x, bins=_grid_bins(x, bins), color="steelblue", alpha=0.85)
    ax_x.set_ylabel("count", fontsize=9); ax_x.tick_params(labelbottom=False); ax_x.grid(alpha=0.3)
    ax_y.hist(y, bins=_grid_bins(y, bins), orientation="horizontal", color="indianred", alpha=0.85)
    ax_y.set_xlabel("count", fontsize=9); ax_y.tick_params(labelleft=False); ax_y.grid(alpha=0.3)
    fig.suptitle(f"{xlabel} vs {ylabel}   (N={x.size})", fontsize=11)
    fig.savefig(outpath, dpi=130); plt.close(fig)


def inputs_vs_target_arrays(backend, target, discard, thin):
    '''(chain, target values, log-prob) for one backend, flattened post discard/thin.
    Returns None if the backend has no blob called `target`. Split out from
    :func:`plot_inputs_vs_target` so several ensembles can be pooled into one figure.'''
    try:
        blobs = backend.get_blobs(discard=discard, thin=thin, flat=True)
    except Exception:
        blobs = None
    names = blobs.dtype.names if (blobs is not None and blobs.dtype.names) else None
    if not names or target not in names:
        return None
    return (backend.get_chain(discard=discard, thin=thin, flat=True),
            np.asarray(blobs[target], dtype=float),
            backend.get_log_prob(discard=discard, thin=thin, flat=True))


def plot_inputs_vs_target(backend, params, target, outdir, tag,
                          discard, thin, bins=50, max_scatter=200000):
    '''One scatter+histogram figure per free input dim (chain columns `params`)
    against the target observable (blob field `target`), coloured by log-prob.
    Reads chain + blobs + log_prob from the HDF5 backend (post discard/thin).
    Returns the list of files written.'''
    got = inputs_vs_target_arrays(backend, target, discard, thin)
    if got is None:
        _emit(f"no blob '{target}' in backend; skipping input-vs-target plots ({tag}).", warn=True)
        return []
    chain, y, logp = got
    return plot_inputs_vs_target_data(chain, y, logp, params, target, outdir, tag,
                                      bins=bins, max_scatter=max_scatter)


def plot_inputs_vs_target_data(chain, y, logp, params, target, outdir, tag,
                               bins=50, max_scatter=200000):
    '''As :func:`plot_inputs_vs_target`, but from already-loaded arrays — so the
    same figures can be made for a single ensemble or for every ensemble pooled.'''
    valid = np.isfinite(y) & (y > 0)         # drop failed / sentinel (-1) values
    rng = np.random.default_rng(0)
    written = []
    for j, name in enumerate(params):
        x = chain[:, j].astype(float)
        m = valid & np.isfinite(x)
        xj, yj = x[m], y[m]
        lpj = logp[m] if logp is not None else None
        if xj.size < 2:
            continue
        idx = (rng.choice(xj.size, max_scatter, replace=False)
               if xj.size > max_scatter else slice(None))
        out = os.path.join(outdir, f"{name}_vs_{target}_{tag}.png")
        _joint_plot(xj, name, yj, target, lpj, idx, out, bins=bins)
        written.append(out)
    _emit(f"wrote {len(written)} input-vs-{target} plot(s) ({tag}) -> {outdir}")
    return written


# ── sign folds (shared by the R-hat and rank checks) ────────────────────────

def _folded_blocks(chains_list, params, discard, thin, canon=False, mag=False):
    '''Slice each ensemble post-burn-in and apply the requested sign fold.

    Returns a list of (ndraw, nwalkers, ndim) float arrays, one per ensemble. Shared
    by rhat_ess and rank_uniformity so the two always analyse identical coordinates
    -- the rank check is only interpretable next to the R-hat it accompanies.

    See rhat_ess for what canon/mag mean physically. NB the copy() below is
    load-bearing, not defensive: a float64 slice comes back as a VIEW, and the
    caller's `chains` must stay signed for what runs afterwards (mode occupancy
    re-reads the raw signs, and abs() is not reversible).'''
    ci = {n: params.index(n) for n in ("M_1", "M_2", "mu") if n in params}
    do_canon = (canon or mag) and len(ci) == 3
    do_mag = mag and len(ci) == 3
    blocks = []
    for ch in chains_list:
        blk = np.asarray(ch[discard::thin], dtype=float)   # (ndraw, nwalkers, ndim)
        if do_canon:
            blk = blk.copy()
            neg = blk[:, :, ci["mu"]] < 0
            for n in ("M_1", "M_2", "mu"):
                blk[:, :, ci[n]] = np.where(neg, -blk[:, :, ci[n]], blk[:, :, ci[n]])
        if do_mag:
            for n in ("M_1", "M_2"):
                blk[:, :, ci[n]] = np.abs(blk[:, :, ci[n]])
        blocks.append(blk)
    return blocks


# ── arviz R-hat / ESS ───────────────────────────────────────────────────────

def rhat_ess(chains_list, params, discard, thin, label, canon=False, mag=False):
    '''chains_list: list of (nsteps, nwalkers, ndim) arrays (one per ensemble).
    Builds an arviz InferenceData with chains = all walkers pooled across the
    given ensembles, then reports R-hat and bulk/tail ESS per parameter.

    canon=True first applies the EXACT overall-flip symmetry (M_1,M_2,mu)->(-,-,-)
    to put every sample in the mu>=0 convention, removing the one *rigorous* 2-fold
    sign label-switching. It deliberately does NOT fold the individual M_1/M_2 signs:
    an individual flip changes Omega by ~15-20% (it is physics, not a symmetry), so
    those stay signed and must converge on their own.

    mag=True does the canonical fold and THEN abs()-es M_1/M_2. That is NOT a
    symmetry -- it collapses the physical sign lobes -- so the result is DIAGNOSTIC
    ONLY and must never gate the verdict. Its value is as a reference floor: read
    against the signed canonical R-hat it separates "the sign lobes are still
    unmixed" (signed R-hat >> magnitude R-hat) from "the magnitudes themselves have
    not converged" (both high).'''
    import arviz as az
    do_mag = mag and len({n for n in ("M_1", "M_2", "mu") if n in params}) == 3
    blocks = _folded_blocks(chains_list, params, discard, thin, canon=canon, mag=mag)
    posterior = {}
    for pi, name in enumerate(params):
        cols = [b[:, :, pi].T for b in blocks]             # each (nwalkers, ndraw)
        min_draw = min(c.shape[1] for c in cols)
        posterior[name] = np.concatenate([c[:, :min_draw] for c in cols], axis=0)
    idata = az.from_dict(posterior=posterior)
    rhat = az.rhat(idata)
    ess_bulk = az.ess(idata, method="bulk")
    ess_tail = az.ess(idata, method="tail")
    n_chains = posterior[params[0]].shape[0]
    n_draws = posterior[params[0]].shape[1]

    total = max(n_chains * n_draws, 1)   # pooled post-burn-in samples (ESS/N efficiency)
    rows = [f"R-hat / ESS ({label}): {n_chains} chains x {n_draws} draws "
            f"(discard={discard}, thin={thin})",
            f"  {'param':10s} {'R-hat':>8s} {'ESS_bulk':>10s} {'ESS_tail':>10s} {'ESS/N':>8s}  flag"]
    # Collect first, aggregate NaN-safely after. Folding with max()/min() as we go
    # SWALLOWS NaNs: max(0.0, nan) is 0.0, because `nan > 0.0` is False, and likewise
    # min(inf, nan) is inf. A chain too short for arviz to estimate anything therefore
    # reported "worst R-hat = 0.000 (converged)" with "min ESS_bulk = inf", and the
    # criteria table read the 0.000 as a PASS -- its `isfinite` guard never saw a NaN.
    rs, es, ts = [], [], []
    for name in params:
        r = float(rhat[name].values)
        eb = float(ess_bulk[name].values)
        et = float(ess_tail[name].values)
        rs.append(r); es.append(eb); ts.append(et)
        flag = ("n/a" if not (np.isfinite(r) and np.isfinite(eb)) else
                "BAD" if (r > 1.1 or eb < 100) else
                ("borderline" if (r > RHAT_MAX or eb < ESS_MIN) else "ok"))
        rows.append(f"  {name:10s} {r:8.3f} {eb:10.0f} {et:10.0f} {eb/total:7.1%}  {flag}")
    worst_r = float(np.nanmax(rs)) if np.any(np.isfinite(rs)) else float("nan")
    worst_e = float(np.nanmin(es)) if np.any(np.isfinite(es)) else float("nan")
    worst_t = float(np.nanmin(ts)) if np.any(np.isfinite(ts)) else float("nan")
    verdict = ("not estimable" if not np.isfinite(worst_r) else
               "magnitude floor, not a verdict" if do_mag else
               "NOT CONVERGED" if worst_r > 1.1 else
               ("borderline" if worst_r > RHAT_MAX else "converged"))
    rows.append(f"worst R-hat = {'n/a' if not np.isfinite(worst_r) else format(worst_r, '.3f')} "
                f"({verdict}); min ESS_bulk = "
                f"{'n/a' if not np.isfinite(worst_e) else format(worst_e, '.0f')}")
    _emit("\n".join(rows))
    return {"worst_rhat": worst_r, "min_ess_bulk": worst_e,
            "min_ess_tail": worst_t, "label": label}


# ── rank statistic / computational faithfulness ─────────────────────────────

def _rank_shape(counts):
    '''Classify one chain's rank histogram: uniform / U / inverted-U / skewed.

    The SHAPE is the diagnostic content of a rank histogram [6]. Two orthogonal
    contrasts of the bin proportions p_b over midpoints x_b in (-1/2, 1/2):

      location    L = sum_b p_b * x_b            (which end of the pooled range)
      dispersion  Q = sum_b p_b * (x_b^2 - m2)   (extremes vs middle)

    Standardised by the EXACT discrete null moments over the B midpoints, so there is
    no continuous-approximation error from using 1/12 and 1/180.

    Reading them here is NOT the SBC reading. In SBC the chains are refits against
    known truth, so a U-shape means the posterior is too narrow. Here the chains are
    independent MCMC replicates of one posterior, so a U-shape means THIS chain is
    over-dispersed relative to the pooled sample, and an inverted-U means it is
    under-dispersed -- concentrated in a sub-region, e.g. trapped in one sign lobe.

    :returns: ``(label, z_location, z_dispersion)``.
    '''
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    B = counts.size
    if n <= 0 or B < 3:
        return "n/a", float("nan"), float("nan")
    p = counts / n
    x = (np.arange(B) + 0.5) / B - 0.5
    m2 = float((x ** 2).mean())
    q = x ** 2 - m2
    # exact null variances of the two contrasts under a uniform draw over the midpoints
    var_x = float((x ** 2).mean())
    var_q = float((q ** 2).mean())
    if var_x <= 0 or var_q <= 0:
        return "n/a", float("nan"), float("nan")
    z_loc = float((p * x).sum() / np.sqrt(var_x / n))
    z_dis = float((p * q).sum() / np.sqrt(var_q / n))
    if max(abs(z_loc), abs(z_dis)) < RANK_Z_FLAT:
        return "uniform", z_loc, z_dis
    if abs(z_loc) >= RANK_Z_FLAT and abs(z_dis) >= RANK_Z_FLAT:
        return "mixed", z_loc, z_dis
    if abs(z_loc) >= abs(z_dis):
        return ("skewed-high" if z_loc > 0 else "skewed-low"), z_loc, z_dis
    return ("U-shaped" if z_dis > 0 else "inverted-U"), z_loc, z_dis


def _holm(pvals):
    '''Holm-Bonferroni step-down adjusted p-values (same order as the input).

    The check runs over every parameter, so the smallest raw p out of ndim is not a
    fair summary: with 9 independent parameters at a true 1% level, one hit is
    expected ~9% of the time. Holm controls the family-wise rate without assuming
    independence, and is uniformly less conservative than plain Bonferroni.'''
    p = np.asarray(pvals, dtype=float)
    m = p.size
    out = np.full(m, np.nan)
    finite = np.where(np.isfinite(p))[0]
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for i, idx in enumerate(order):
        adj = min(1.0, (order.size - i) * float(p[idx]))
        running = max(running, adj)          # enforce monotonicity
        out[idx] = running
    return out


def rank_uniformity(chains_list, params, discard, max_tau, label,
                    canon=True, segments=RANK_SEGMENTS):
    '''Rank-statistic uniformity across chains -- the computational-faithfulness check.

    Rank every draw of a parameter POOLED over all chains, then histogram the ranks
    separately per chain. If all chains target the same distribution each histogram is
    uniform; departures show HOW they disagree (see _rank_shape). This is the rank plot
    of Vehtari et al. [4], the computable stand-in for the simulation-based calibration
    rank statistic of [6].

    IT IS NOT SBC. SBC ranks a known simulating parameter inside posteriors refitted to
    simulated data, and so certifies calibration; that needs hundreds of refits and is
    out of reach for a likelihood that runs the full physics pipeline. This check only
    certifies that the chains agree WITH EACH OTHER. A uniform result is evidence
    against non-convergence, never evidence that the posterior is correctly calibrated.

    Chains are the ENSEMBLES, not the walkers: emcee's DE/snooker moves couple walkers
    within an ensemble, so walker agreement is not evidence of convergence. With a
    single ensemble there are no independent replicates, so it is split into `segments`
    consecutive time slices instead -- that is a stationarity/drift check, and is
    labelled as such rather than passed off as the cross-seed one.

    The histogram uses EVERY post-burn-in draw. The test does not: with tau ~ 2000 the
    draws are far from independent, and a chi-square that assumes independence rejects
    essentially always (measured: statistic inflated ~17x, 100% false-positive rate).
    Scaling by ESS/N over-corrects by an order of magnitude, because bin counts are sums
    of INDICATORS whose autocorrelation time is much shorter than the parameter's. So the
    test thins by max_tau to genuinely independent draws and uses an exact
    contingency-table homogeneity chi-square: pooled ranks fix both margins (N per chain,
    CN/B per bin), giving dof = (C-1)(B-1), which is calibrated at its nominal level.

    :returns: dict with worst_p (Holm-adjusted), worst_param, per_param, hist (for the
        plots, from all draws), thinned_hist, shapes, n_chains, n_thinned, label,
        segmented.
    '''
    from scipy.stats import chi2, rankdata

    segmented = len(chains_list) < 2
    blocks = _folded_blocks(chains_list, params, discard, 1, canon=canon)
    if segmented:
        # one ensemble -> consecutive time slices as pseudo-chains
        blk = blocks[0]
        cut = (blk.shape[0] // segments) * segments
        groups = [blk[i * cut // segments:(i + 1) * cut // segments] for i in range(segments)]
    else:
        groups = blocks
    C = len(groups)

    # Flatten each chain over its walkers: one row = one draw = one unit of weight.
    flat = [g.reshape(-1, g.shape[2]) for g in groups]
    n_all = min(f.shape[0] for f in flat)          # equal lengths -> both margins fixed
    flat = [f[:n_all] for f in flat]

    # Independent subsample for the TEST: thin in TIME by tau and keep every walker.
    # Striding the flattened array by tau*nwalkers instead would land on the same walker
    # every time and throw away all the others -- on a 256-walker run that leaves single
    # digits of draws and the test goes unestimable. Treating the surviving walkers as
    # independent once a full tau has elapsed is emcee's own ESS convention
    # (ESS ~ nwalkers * nsteps / tau, [2]); it is a convention, not a theorem, since the
    # DE/snooker moves do couple walkers, so read the p-value alongside R-hat rather than
    # as an independent verdict.
    step = max(1, int(np.ceil(max_tau))) if np.isfinite(max_tau) else 1
    thin_groups = [g[::step] for g in groups]
    tflat = [g.reshape(-1, g.shape[2]) for g in thin_groups]
    n_thin = min(f.shape[0] for f in tflat) if tflat else 0
    tflat = [f[:n_thin] for f in tflat]

    B = RANK_BINS
    rows = [f"rank statistic / uniformity across chains ({label}): {C} chains x "
            f"{n_all} draws, {B} bins",
            f"  test on {n_thin} tau-independent draws/chain (every {step}th iteration "
            f"~ max_tau={max_tau:.0f}, all walkers kept); histogram uses all draws",
            "  NOT simulation-based calibration: this certifies the chains agree with "
            "each other,",
            "  not that the posterior is calibrated (SBC needs refits against simulated "
            "truth). [4,6]"]

    # chi-square needs an expected count of ~5 per cell to be trustworthy; below that
    # the asymptotic distribution is a poor fit and the p-value means little.
    estimable = n_thin >= 5 * B
    if not estimable:
        rows.append(f"  not estimable: {n_thin} independent draws/chain gives under 5 "
                    f"expected per bin ({5 * B} needed for {B} bins) -- run longer, or "
                    f"lower RANK_BINS")

    hist_all, hist_thin, raw_p, per_param, shapes = {}, {}, [], {}, {}
    for pi, name in enumerate(params):
        cols = [f[:, pi] for f in flat]
        pooled = np.concatenate(cols)
        if not np.all(np.isfinite(pooled)):
            # a non-finite draw makes the ranking meaningless for this parameter; report
            # it as unestimable rather than ranking nan/inf into an arbitrary bin
            hist_all[name] = np.zeros((C, B))
            raw_p.append(float("nan"))
            shapes[name] = ["n/a"] * C
            per_param[name] = {"chi2": float("nan"), "dof": (C - 1) * (B - 1)}
            continue
        r = rankdata(pooled, method="average").reshape(C, n_all)
        edges = np.linspace(0.0, float(r.size), B + 1)
        hist_all[name] = np.stack([np.histogram(r[c], bins=edges)[0] for c in range(C)])
        # Shapes are classified on the THINNED histogram, never this one: _rank_shape
        # standardises by 1/sqrt(n), so at n ~ 4e6 correlated draws every |z| clears 2
        # and every chain would be labelled shaped. Same trap as the chi-square.
        shapes[name] = ["n/a"] * C

        if estimable:
            # rank the thinned draws in their own right, so the test statistic really is
            # over independent observations rather than a subsample of pooled ranks
            tp = np.concatenate([f[:, pi] for f in tflat])
            tr = rankdata(tp, method="average").reshape(C, n_thin)
            tedges = np.linspace(0.0, float(tr.size), B + 1)
            th = np.stack([np.histogram(tr[c], bins=tedges)[0] for c in range(C)])
            hist_thin[name] = th
            shapes[name] = [_rank_shape(th[c])[0] for c in range(C)]
            exp = n_thin / B
            stat = float(((th - exp) ** 2 / exp).sum())
            dof = (C - 1) * (B - 1)
            raw_p.append(float(chi2.sf(stat, dof)))
            per_param[name] = {"chi2": stat, "dof": dof}
        else:
            raw_p.append(float("nan"))
            per_param[name] = {"chi2": float("nan"), "dof": (C - 1) * (B - 1)}

    adj = _holm(raw_p)
    for name, rp, ap in zip(params, raw_p, adj):
        per_param.setdefault(name, {}).update({"p_raw": rp, "p_holm": ap})

    rows.append(f"  {'param':10s} {'chi2':>9s} {'dof':>4s} {'p_raw':>9s} "
                f"{'p_holm':>9s}  shape per chain")
    for name in params:
        d = per_param[name]
        sh = ",".join(shapes[name])
        pr = "n/a" if not np.isfinite(d["p_raw"]) else f"{d['p_raw']:.2e}"
        ph = "n/a" if not np.isfinite(d["p_holm"]) else f"{d['p_holm']:.2e}"
        c2 = "n/a" if not np.isfinite(d["chi2"]) else f"{d['chi2']:9.1f}"
        flag = "" if not np.isfinite(d["p_holm"]) else \
            ("  <- non-uniform" if d["p_holm"] < RANK_P_MIN else "")
        rows.append(f"  {name:10s} {c2:>9s} {d['dof']:4d} {pr:>9s} {ph:>9s}  {sh}{flag}")

    if np.any(np.isfinite(adj)):
        w = int(np.nanargmin(adj))
        worst_p, worst_param = float(adj[w]), params[w]
        verdict = ("non-uniform -- chains disagree" if worst_p < RANK_P_MIN
                   else "uniform (consistent with one distribution)")
        rows.append(f"worst Holm-adjusted p = {worst_p:.2e} ({worst_param}) -> {verdict}")
    else:
        worst_p, worst_param = float("nan"), None
        rows.append("worst Holm-adjusted p = n/a (not estimable)")
    _emit("\n".join(rows))

    return {"worst_p": worst_p, "worst_param": worst_param, "per_param": per_param,
            "hist": hist_all, "thinned_hist": hist_thin, "shapes": shapes,
            "n_chains": C, "n_thinned": n_thin if estimable else 0,
            "label": label, "segmented": segmented}


_RANK_COLOURS = ("#2c6fbb", "#d62728", "#e8a33d", "#2ca02c",
                 "#8c564b", "#9467bd", "#17becf", "#7f7f7f")


def plot_rank_hist(res, params, outpath) -> str:
    '''Rank histograms: one panel per parameter, one step histogram per chain.

    Uniform is the null. The solid curves are the TAU-THINNED sample, matching the
    p-value and the ECDF figure, as the SBC literature recommends [6]: autocorrelation
    does not bias the histogram (it is flat in expectation either way) but it does make
    an all-draws histogram look far smoother than its information content justifies --
    measured ~2x on a tau=199 toy, more at production tau -- because 4 M correlated draws
    carry a few thousand draws' worth of information. Reading precision off that is the
    trap. The faint curves are the all-draws version, kept for reference so the two can
    be compared; where they differ, the solid one is the honest one.'''
    if not any(res["hist"][p].sum() for p in params if p in res["hist"]):
        raise ValueError("no post-burn-in draws to rank (chain shorter than the discard)")
    n = len(params)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)
    C = res["n_chains"]
    thin = res.get("thinned_hist") or {}
    for i, name in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        h_all = res["hist"][name]
        h = thin.get(name, h_all)          # solid = thinned when the test was estimable
        B = h.shape[1]
        centres = (np.arange(B) + 0.5) / B
        for c in range(C):
            col = _RANK_COLOURS[c % len(_RANK_COLOURS)]
            if name in thin:               # faint all-draws reference underneath
                ta = h_all[c].sum()
                if ta:
                    ax.step(centres, h_all[c] / ta * B, where="mid",
                            color=col, lw=0.8, alpha=0.35)
            tot = h[c].sum()
            frac = h[c] / tot if tot else h[c]
            ax.step(centres, frac * B, where="mid", color=col, lw=1.4,
                    label=f"{'seg' if res['segmented'] else 'ens'}{c}")
        ax.axhline(1.0, color="k", ls="--", lw=1.0, alpha=0.6)
        d = res["per_param"].get(name, {})
        ph = d.get("p_holm", float("nan"))
        ptxt = "p=n/a" if not np.isfinite(ph) else f"p={ph:.1e}"
        bad = np.isfinite(ph) and ph < RANK_P_MIN
        shapes = sorted({s for s in res["shapes"].get(name, []) if s not in ("uniform", "n/a")})
        sub = ("  " + ",".join(shapes)) if shapes else ""
        ax.set_title(f"{name}   {ptxt}{sub}", fontsize=9,
                     color="#d62728" if bad else "k")
        ax.set_xlim(0, 1)
        # Zoom on the deviations. Anchoring at 0 wastes ~85% of the axis: the curves live
        # within a few percent of 1, so a 0-based panel squashes exactly what is being
        # judged. Symmetric about 1 so over- and under-density stay visually comparable.
        dev = max(0.05, float(np.nanmax(np.abs(np.concatenate(
            [h[c] / h[c].sum() * B - 1.0 for c in range(C) if h[c].sum()])))))
        ax.set_ylim(1 - 1.15 * dev, 1 + 1.15 * dev)
        ax.set_xlabel("normalised rank", fontsize=8)
        ax.set_ylabel("density (1 = uniform)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=7, ncol=2, frameon=False)
    nthin = res.get("n_thinned", 0)
    which = (f"solid = {nthin} tau-independent draws/chain, faint = all draws"
             if nthin and thin else "all draws (test not estimable)")
    fig.suptitle(f"rank statistic ({res['label']}) -- uniform if all chains sample the "
                 f"same distribution; NOT simulation-based calibration\n{which}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return outpath


def plot_rank_ecdf(res, params, outpath) -> str:
    '''Rank ECDF minus uniform, with a simultaneous confidence band [7].

    The honest read of pass/fail: a curve leaving the band is a real departure. Built
    from the TAU-THINNED sample, because the band construction assumes independent
    draws -- fed all draws it would be ~sqrt(tau) too narrow and condemn everything.'''
    from arviz.stats.ecdf_utils import ecdf_confidence_band

    thin = res.get("thinned_hist") or {}
    usable = [p for p in params if p in thin]
    if not usable:
        raise ValueError("no tau-independent sample: rank ECDF needs an estimable test")
    n = len(usable)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)
    C, ndraw = res["n_chains"], res["n_thinned"]
    ev = np.linspace(0, 1, 101)
    try:
        lo, hi = ecdf_confidence_band(ndraw, ev, ev, prob=0.95, method="optimized")
    except Exception:                      # optimiser can fail on small ndraw
        lo, hi = ecdf_confidence_band(ndraw, ev, ev, prob=0.95, method="pointwise")
    for i, name in enumerate(usable):
        ax = axes[i // ncol][i % ncol]
        ax.fill_between(ev, lo - ev, hi - ev, color="#6b6b6b", alpha=0.25,
                        label="95% simultaneous" if i == 0 else None)
        h = thin[name]
        B = h.shape[1]
        edges = np.arange(B + 1) / B
        for c in range(C):
            tot = h[c].sum()
            ecdf = np.concatenate([[0.0], np.cumsum(h[c]) / tot]) if tot else np.zeros(B + 1)
            ax.step(edges, ecdf - edges, where="post",
                    color=_RANK_COLOURS[c % len(_RANK_COLOURS)], lw=1.3,
                    label=f"{'seg' if res['segmented'] else 'ens'}{c}" if i == 0 else None)
        ax.axhline(0.0, color="k", ls="--", lw=1.0, alpha=0.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("normalised rank", fontsize=8)
        ax.set_ylabel("ECDF - uniform", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(f"rank ECDF vs uniform ({res['label']}) -- {ndraw} tau-independent "
                 f"draws/chain, 95% simultaneous band", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return outpath


def _emit_mode_occupancy(chains, params, discard, thin):
    '''Per-ensemble occupancy of (a) the LSP-type proxy argmin(|M_1|,|M_2|,|mu|) =
    bino/wino/higgsino, and (b) the PHYSICAL relative-sign octants (sign M_1, sign M_2
    in the mu>=0 convention -- these are real modes, not a symmetry). The cross-seed
    spread (max-min across ensembles) is the honest multimodal-convergence signal:
    independent seeds must agree on the fractions. Both are proxies, fuzzy near
    mixed-neutralino boundaries -- report, do not over-read.'''
    need = ("M_1", "M_2", "mu")
    if not all(n in params for n in need):
        return
    iM1, iM2, imu = (params.index(n) for n in need)
    lsp, oct4 = [], []
    for ch in chains:
        blk = np.asarray(ch[discard::thin], dtype=float).reshape(-1, len(params))
        a = np.abs(blk[:, [iM1, iM2, imu]])
        lst = np.argmin(a, axis=1)
        lsp.append([float(np.mean(lst == i)) for i in range(3)])
        neg = blk[:, imu] < 0                              # mu>=0 canonical relative signs
        s1 = np.where(neg, -blk[:, iM1], blk[:, iM1])
        s2 = np.where(neg, -blk[:, iM2], blk[:, iM2])
        oct4.append([float(np.mean((np.sign(s1) == a1) & (np.sign(s2) == a2)))
                     for a1 in (1, -1) for a2 in (1, -1)])

    def _tab(title, names, fracs):
        rows = [title, "  " + "".join(f"{n:>14s}" for n in names)]
        for k, fr in enumerate(fracs):
            rows.append(f"  ens {k:<3d}" + "".join(f"{100*x:13.1f}%" for x in fr))
        if len(fracs) > 1:
            sp = np.array(fracs).max(0) - np.array(fracs).min(0)
            rows.append("  spread " + "".join(f"{100*x:13.1f}%" for x in sp)
                        + "  <- cross-seed disagreement (small = converged occupancy)")
        _emit("\n".join(rows))

    _tab("LSP-type occupancy proxy = argmin(|M_1|,|M_2|,|mu|):",
         ["bino|M_1|", "wino|M_2|", "higgsino|mu|"], lsp)
    _tab("relative-sign octant occupancy (mu>=0 convention; PHYSICAL modes):",
         ["M1+M2+", "M1+M2-", "M1-M2+", "M1-M2-"], oct4)


# ── ntuple bridge ────────────────────────────────────────────────────────────

def _from_emcee_ntuple(*names):
    '''Import symbol(s) from the shared Run3ModelGen.emcee_ntuple module — the
    single source of truth for both the HDF5 -> ROOT ntuple layout (h5_to_ntuple)
    and the burn-in policy (resolve_burnin), so the diagnostics can never drift
    from the sampler. Imported lazily so the rest of the diagnostics stays
    package-independent; if the package isn't importable (diagnostics run without
    sourcing build/setup.sh), add the built package dir next to this script and
    retry. Returns a single object for one name, else a tuple.'''
    try:
        import Run3ModelGen.emcee_ntuple as m
    except ImportError:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))   # source/Run3ModelGen/scripts
        repo = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        sys.path.insert(0, os.path.join(repo, "build"))     # build/Run3ModelGen (symlinked pkg)
        import Run3ModelGen.emcee_ntuple as m
    got = tuple(getattr(m, n) for n in names)
    return got[0] if len(got) == 1 else got


# ── main ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", nargs="+", default=None, help="One or more HDF5 backend paths.")
    p.add_argument("--glob", default=None, help="Glob pattern for backend paths.")
    p.add_argument("--outdir", default="emcee_diag", help="Directory for plots/output.")
    p.add_argument("--params", default=None, help="Comma-separated param names (default: from meta free_names).")
    p.add_argument("--discard", default="auto",
                   help="Burn-in iterations to discard. 'auto' = the sampler's rule "
                        "max(burn_in, ceil(burn_in_factor*max_tau)) from the sidecar "
                        "(burn_in_factor defaults to 3), so the diagnostics match the ntuple.")
    p.add_argument("--thin", default="auto",
                   help="Thinning. 'auto' = max(1, floor(min_tau/2)).")
    p.add_argument("--to-ntuple", default=None, help="Write one ROOT ntuple per backend into this dir.")
    p.add_argument("--target", default=None,
                   help="Observable (blob field) for the input-vs-target plots "
                        "(default: auto-detect the constrained observable, e.g. m_h / Omega).")
    return p.parse_args()


def main():
    args = _parse_args()
    _REPORT.clear()
    paths = list(args.backend or [])
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    if not paths:
        raise SystemExit("no backends: pass --backend and/or --glob")

    os.makedirs(args.outdir, exist_ok=True)
    try:
        _diagnose(args, paths)
    finally:
        # Always emit the plain-text report — even if a diagnostic step raised — so a
        # partial/failed run still leaves whatever criteria + verdict were captured on
        # disk (the chain rule runs this best-effort as its final step).
        txt_path = os.path.join(args.outdir, "diagnostics.txt")
        try:
            with open(txt_path, "w") as fh:
                fh.write("\n".join(_REPORT) + "\n")
            log.info(f"diagnostics report -> {txt_path}")
        except OSError as exc:
            log.warning(f"could not write {txt_path}: {exc}")


def _diagnose(args, paths):
    ens = [load_backend(p) for p in paths]

    # Parameter names from the first sidecar (fallback to generic names).
    if args.params:
        params = [s.strip() for s in args.params.split(",") if s.strip()]
    else:
        params = ens[0]["meta"].get("free_names")
        if not params:
            ndim = ens[0]["backend"].get_chain().shape[2]
            params = [f"p{i}" for i in range(ndim)]

    _emit(f"{len(ens)} ensemble(s); params={params}")
    for e in ens:
        _emit(f"  {e['path']}: iterations={e['iteration']} "
              f"nwalkers={e['meta'].get('nwalkers', '?')}")

    # Per-ensemble: acceptance, tau, and single-ensemble plots.
    chains = []
    all_tau_max = []
    all_tau_min = []
    sug_burnins = []
    n50_flags = []           # N > 50*tau per ensemble (for the convergence verdict)
    acc_means = []           # mean acceptance per ensemble
    for i, e in enumerate(ens):
        b = e["backend"]
        chain = b.get_chain()  # (nsteps, nwalkers, ndim)
        chains.append(chain)
        nwalkers, ndim = chain.shape[1], chain.shape[2]

        # acceptance fraction via a no-op sampler bound to the backend
        try:
            s = emcee.EnsembleSampler(nwalkers, ndim, lambda x: 0.0, backend=b)
            acc = s.acceptance_fraction
            acc_mean = float(np.mean(acc))
        except Exception:
            acc_mean = float("nan")
        acc_means.append(acc_mean)

        tau, ok = autocorr_time(chain, tol=TAU_FACTOR)
        n50_flags.append(bool(ok))
        all_tau_max.append(np.nanmax(tau)); all_tau_min.append(np.nanmin(tau))
        rows = [f"ensemble {i} ({e['path']}):",
                f"  mean acceptance = {acc_mean:.3f}  (healthy ~{ACC_BAND[0]}-{ACC_BAND[1]})",
                f"  N > {TAU_FACTOR}*tau: {'YES' if ok else 'NO (tau is a lower bound)'}  "
                f"(N={chain.shape[0]}, max_tau={np.nanmax(tau):.1f})",
                f"  {'param':10s} {'tau':>8s}"]
        for p in range(ndim):
            rows.append(f"  {params[p]:10s} {tau[p]:8.1f}")
        sug = suggested_burnin_from_logprob(b.get_log_prob())
        sug_burnins.append(sug)
        rows.append(f"  suggested burn-in (log-prob plateau) ~ {sug} iters")
        _emit("\n".join(rows))

        tag = f"ens{i}"
        plot_traces(chain, params, os.path.join(args.outdir, f"trace_{tag}.png"))
        if chain.shape[0] > 20:
            plot_tau_vs_n(chain, params, os.path.join(args.outdir, f"tau_vs_n_{tag}.png"))

    # discard / thin selection. Burn-in uses the SAME rule as the sampler's
    # ntuple (resolve_burnin, keyed off the sidecar meta), so the ntuple and
    # these diagnostics discard the same number of iterations.
    max_tau = float(np.nanmax(all_tau_max))
    min_tau = float(np.nanmin(all_tau_min))
    min_iter = min(c.shape[0] for c in chains)
    meta0 = ens[0]["meta"]
    burn_in_floor = int(meta0.get("burn_in", 0) or 0)
    burn_in_factor = float(meta0.get("burn_in_factor", 3.0))
    if args.discard == "auto":
        discard = resolve_burnin(max_tau, min_iter, burn_in_floor, burn_in_factor)
    else:
        discard = int(args.discard)
    if args.thin == "auto":
        thin = max(1, int(0.5 * min_tau)) if np.isfinite(min_tau) else 1
    else:
        thin = max(1, int(args.thin))
    discard = max(0, min(discard, min_iter - 2))
    _emit(f"using discard={discard}, thin={thin}  "
          f"(rule: max(burn_in={burn_in_floor}, {burn_in_factor:g}*max_tau); "
          f"max_tau={max_tau:.1f}, min_tau={min_tau:.1f})")

    # Is the burn-in adequate? Compare the discard we will apply against the
    # log-prob-plateau estimate (worst over ensembles).
    sug_max = max(sug_burnins) if sug_burnins else 0
    bi_verdict = "OK" if discard >= sug_max else "TOO SHORT -- raise burn_in / burn_in_factor"
    _emit(f"burn-in check: log-prob plateau ~ {sug_max} iters vs "
          f"discard {discard}  ->  {bi_verdict}")

    # Per-ensemble corner + input-vs-target plots (post burn-in), one set per
    # chain. The target observable comes from the first meta (same config for all).
    target = infer_target_obs(ens[0]["meta"], override=args.target)
    if not target:
        _emit("no target observable identified; "
              "skipping input-vs-target plots (pass --target).", warn=True)
    # Plots use ALL post-burn-in samples (thin=1), not the R-hat/ESS thinning —
    # thinning is for independent-sample *estimates*, not for visual density.
    # A subsample cap keeps the corner fast on multi-million-sample chains.
    rng_plot = np.random.default_rng(0)
    MAX_CORNER = 200000
    for i, e in enumerate(ens):
        tag = f"ens{i}"
        # target-observable (e.g. Omega) marginal for the corner's upper-right inset
        tvals = None
        if target:
            try:
                bl = e["backend"].get_blobs(discard=discard, flat=True)
                if bl is not None and bl.dtype.names and target in bl.dtype.names:
                    tv = np.asarray(bl[target], float)
                    tvals = tv[np.isfinite(tv) & (tv > 0)]
            except Exception:
                tvals = None
        try:
            flat = chains[i][discard:].reshape(-1, chains[i].shape[2])
            if flat.shape[0] > MAX_CORNER:
                flat = flat[rng_plot.choice(flat.shape[0], MAX_CORNER, replace=False)]
            if flat.shape[0] > 10:
                plot_corner(flat, params, os.path.join(args.outdir, f"corner_{tag}.png"),
                            target_vals=tvals, target_name=target)
        except Exception as exc:
            _emit(f"corner plot ({tag}) skipped: {exc}", warn=True)
        try:
            plot_logprob(e["backend"], os.path.join(args.outdir, f"logprob_{tag}.png"),
                         discard=discard)
        except Exception as exc:
            _emit(f"logprob plot ({tag}) skipped: {exc}", warn=True)
        if target:
            try:
                plot_inputs_vs_target(e["backend"], params, target,
                                      args.outdir, tag, discard, thin=1)
            except Exception as exc:
                _emit(f"input-vs-target plots ({tag}) skipped: {exc}", warn=True)

    # ── every ensemble pooled ────────────────────────────────────────────────
    # The R-hat/ESS above are already cross-ensemble; these are the matching
    # PICTURES, so the run has one posterior to look at rather than N to compare by
    # eye. Pooling is legitimate because the stored chain is already a correctly
    # weighted sample: one row = one unit of weight (a point a walker lingered on
    # appears once per step it stayed), so concatenating ensembles just adds weight.
    # Only meaningful with >1 ensemble; with one, the ens0 figures already are this.
    #
    # Uniformly subsampled to POOL_MAX rows total: a uniform draw preserves the
    # weighting exactly, and four full ensembles of a 10k-model run is ~1.6 GB of
    # chain to hold at once.
    POOL_MAX = 2000000
    if len(ens) > 1:
        tag = "all"
        try:
            flats = [chains[i][discard:].reshape(-1, chains[i].shape[2])
                     for i in range(len(ens))]
            pooled = np.concatenate(flats, axis=0)
            if pooled.shape[0] > POOL_MAX:
                pooled = pooled[rng_plot.choice(pooled.shape[0], POOL_MAX, replace=False)]
            tv_all = None
            if target:
                tvs = []
                for e in ens:
                    try:
                        bl = e["backend"].get_blobs(discard=discard, flat=True)
                        if bl is not None and bl.dtype.names and target in bl.dtype.names:
                            tvs.append(np.asarray(bl[target], float))
                    except Exception:
                        pass
                if tvs:
                    tv = np.concatenate(tvs)
                    tv_all = tv[np.isfinite(tv) & (tv > 0)]
            corner_in = pooled
            if corner_in.shape[0] > MAX_CORNER:
                corner_in = corner_in[rng_plot.choice(corner_in.shape[0], MAX_CORNER,
                                                      replace=False)]
            if corner_in.shape[0] > 10:
                plot_corner(corner_in, params,
                            os.path.join(args.outdir, f"corner_{tag}.png"),
                            target_vals=tv_all, target_name=target)
        except Exception as exc:
            _emit(f"combined corner plot skipped: {exc}", warn=True)
        if target:
            try:
                parts = [inputs_vs_target_arrays(e["backend"], target, discard, 1)
                         for e in ens]
                parts = [p for p in parts if p is not None]
                if parts:
                    ch = np.concatenate([p[0] for p in parts], axis=0)
                    yy = np.concatenate([p[1] for p in parts])
                    lp = ([p[2] for p in parts] if all(p[2] is not None for p in parts)
                          else None)
                    lp = np.concatenate(lp) if lp is not None else None
                    if ch.shape[0] > POOL_MAX:
                        sel = rng_plot.choice(ch.shape[0], POOL_MAX, replace=False)
                        ch, yy = ch[sel], yy[sel]
                        lp = lp[sel] if lp is not None else None
                    plot_inputs_vs_target_data(ch, yy, lp, params, target,
                                               args.outdir, tag)
                else:
                    _emit(f"no ensemble carries blob '{target}'; no combined "
                          f"input-vs-target plots.", warn=True)
            except Exception as exc:
                _emit(f"combined input-vs-target plots skipped: {exc}", warn=True)

    # R-hat / ESS / occupancy are computed on the FULL, UNTHINNED post-burn-in chain
    # (thin=1). arviz de-correlates internally, so thinning would only throw away
    # information and destabilise the estimate (Vehtari et al. 2021). The tau-scaled
    # `thin` is used ONLY for the --to-ntuple product, never for these estimates.
    # NOTE: peak memory scales with the unthinned draw count -- the cross-ensemble
    # arviz object is ~ nseeds * nwalkers * (min_iter - discard) * ndim * 8 bytes
    # (a few GB for this run); size the diagnostics job's --mem accordingly.
    thin_rhat = 1
    _emit(f"R-hat / ESS / occupancy on the UNTHINNED post-burn-in chain "
          f"(thin=1; ~{max(0, min_iter - discard)} draws/walker)")
    rhat_res = None
    try:
        canon_ok = all(n in params for n in ("M_1", "M_2", "mu"))
        single = len(chains) == 1
        # RAW -- exposes the sign label-switching inflation.
        r = rhat_ess([chains[0]], params, discard, thin_rhat,
                     label="within ensemble 0 (raw, walkers as chains)")
        if not single:
            r = rhat_ess(chains, params, discard, thin_rhat,
                         label=f"cross-ensemble ({len(chains)} seeds, raw)")
        rhat_res = r
        # EXACT overall-flip fold (mu>=0). Only ONE overall sign is convention; the
        # individual M_1/M_2 signs are physical multimodality and stay signed.
        if canon_ok:
            rc = rhat_ess([chains[0]], params, discard, thin_rhat,
                          label="within ensemble 0 (mu>=0 canonical, exact fold)", canon=True)
            if not single:
                rc = rhat_ess(chains, params, discard, thin_rhat,
                              label=f"cross-ensemble ({len(chains)} seeds, mu>=0 canonical)", canon=True)
            rhat_res = rc     # gate on the exact-fold result
            # Reference floor only -- deliberately NOT assigned to rhat_res. abs()-ing
            # M_1/M_2 collapses the physical sign lobes, so the gap between the signed
            # result above and this one IS the unmixed-sign-lobe penalty: signed R-hat
            # falling toward this floor over successive runs = the sign moves are
            # working; the floor itself staying high = the magnitudes have not converged.
            scope = "within ensemble 0" if single else f"cross-ensemble ({len(chains)} seeds)"
            rhat_ess(chains, params, discard, thin_rhat, mag=True,
                     label=f"{scope}, |M_1|,|M_2| magnitude fold -- DIAGNOSTIC ONLY, "
                           f"sign is physical; NOT a convergence certificate")
        _emit_mode_occupancy(chains, params, discard, thin_rhat)
    except Exception as exc:
        _emit(f"arviz R-hat/ESS skipped: {exc}", warn=True)

    # Rank statistic (computational faithfulness) on ALL samples pooled. Same coordinates
    # as the gating R-hat above -- the canonical fold -- so the two are read together:
    # in raw coordinates the exact (M_1,M_2,mu) flip symmetry alone would force
    # non-uniformity that means nothing.
    rank_res = None
    try:
        canon_rank = all(n in params for n in ("M_1", "M_2", "mu"))
        scope = ("time segments of one ensemble" if len(chains) == 1
                 else f"{len(chains)} ensembles")
        rank_res = rank_uniformity(chains, params, discard, max_tau,
                                   label=f"all samples, {scope}"
                                         + (", mu>=0 canonical" if canon_rank else ""),
                                   canon=canon_rank)
        for fn, base in ((plot_rank_hist, "rank_all.png"),
                         (plot_rank_ecdf, "rank_ecdf_all.png")):
            try:
                _emit(f"wrote {os.path.basename(fn(rank_res, params, os.path.join(args.outdir, base)))}"
                      f" -> {args.outdir}")
            except Exception as exc:
                _emit(f"{base} skipped: {exc}", warn=True)
    except Exception as exc:
        _emit(f"rank uniformity skipped: {exc}", warn=True)

    # Every criterion spelled out (requirement / why / this scan), then the verdict.
    _emit_criteria_table(rhat_res, n50_flags, [c.shape[0] for c in chains], all_tau_max,
                         acc_means, discard, sug_max, rank_res=rank_res)
    _emit_convergence_verdict(rhat_res, n50_flags, acc_means, max_tau)

    # optional ntuple bridge (shared writer; the only part that needs the package)
    if args.to_ntuple:
        h5_to_ntuple = _from_emcee_ntuple("h5_to_ntuple")
        os.makedirs(args.to_ntuple, exist_ok=True)
        for i, e in enumerate(ens):
            if not e["meta"]:
                _emit(f"no meta for {e['path']}; skipping ntuple.", warn=True)
                continue
            out = os.path.join(args.to_ntuple, f"ntuple.{i}.root")
            n, nbr = h5_to_ntuple(e["path"], out, discard=discard, thin=thin, meta=e["meta"])
            _emit(f"wrote {n} samples -> {out} ({nbr} branches)")

    _emit(f"plots written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
