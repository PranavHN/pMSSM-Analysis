# MCMC scan — emcee sampler

`prior: mcmc` runs the [emcee](https://emcee.readthedocs.io/en/stable/)
affine-invariant ensemble sampler (`python/emcee_sampler.py`). It runs as one
opaque **chain** job — the whole pipeline lives inside sampling, so it can't be
split into per-tool rules (see `workflow/rules/chain.smk`). The likelihood is the
`constraints:` block (→ chi2) and the physics pipeline is the usual
`ModelGenerator` + `steps.py`.

**Constraint types** (`_compute_chi2_term`). `gaussian`, `upper_bound` and
`lower_bound` are χ² *penalties* on one observable against `central`/`uncertainty`.
`hard_window` is a **veto**: 0 inside, `inf` outside, strict bounds, `lo`/`hi` each
optional — so it expresses a floor (`lo: 0`, e.g. `m_h > 0` to reject SPheno's `-1`
sentinel), a window (`lo` + `hi`) or a boolean flag (`lo: 0.5` on a 0/1 observable
such as micromegas' `cdm_is_neutralino`). It needs no `central`/`uncertainty`.

The distinction matters because **a χ² term cannot veto, only penalise**: a sentinel
`Omega = -1` scored against `0.120 ± 0.012` gives a finite χ² ≈ 8711, so it is merely
improbable, and a walker with nowhere better to go can sit on it. Mixing the two is
the intended use — the penalties set the *shape* of the posterior, the vetoes set its
*support*. Beware the degenerate case: with **only** `hard_window` constraints the
likelihood is an indicator function, the target is uniform on the allowed region,
`emcee_logp` is identically 0, and every marginal becomes a parameter-space *volume*
measure rather than a fit (which also makes the log-prob trace plot and the
log-prob-coloured scatter plots useless). See
`run/relic_slurm/SUBMISSION_neutralino_v4.md` for a measured side-by-side.

## Why emcee

The previous in-house Metropolis-Hastings sampler (since removed) used a
per-parameter Gaussian random walk with reflecting boundaries and a single scalar
width-adaptation. On the correlated, thin relic-density isosurface that needed
heavy manual proposal-width tuning and still mixed poorly (a documented run
reached split-R̂ ≈ 38, ESS ≈ 50 over 410k samples). emcee's
**stretch move is affine-invariant**: it adapts automatically to the local scale
and linear correlations of the target, so no proposal-width tuning is required.

## How it works

* The `chain` rule body is `workflow/scripts/r3mg_batch.py::chain()`, which calls
  `run_emcee_scan(...)` (config in, ROOT ntuple + `chain.json` manifest out; the
  manifest records `engine: emcee`).
* The parameter vector → (chi2, observables) mapping is
  `ModelGenerator.evaluate_theta`, which reuses `_run_and_evaluate` (steps + chi2
  from `constraints`) and `_extract_observable`. The log-probability is
  `-chi2/2 + flat_prior`; out-of-bounds and pipeline-failed points return `-inf`.
* Evaluation is parallelised with a `multiprocessing.Pool` of `mcmc.nworkers`.
  Each worker builds its **own** `ModelGenerator` in an isolated scratch dir (under
  the ephemeral job tempdir, not the scan dir) and `chdir`s into it, so concurrent
  `os.system` calls (e.g. superiso's `output.flha`) never collide.
* The `chain` rule requests `mcmc.nworkers` CPUs (its `threads` are wired from the
  config); BLAS is pinned to one thread in `r3mg_batch.py`, so the pool never
  oversubscribes. For a whole-node run set `nworkers` to the node's core count.
* Observables are recorded per sample via emcee **blobs** (`mcmc.record_observables`).
* Output is an `emcee.backends.HDFBackend` (`<scan_dir>/chain.h5`) plus a sidecar
  `chain.h5.meta.json`, and the ROOT `"susy"` ntuple. `chain.h5` is **side-state,
  not a declared rule output**, so Snakemake never deletes it and the emcee path
  never wipes the scan dir — a re-run **resumes** the chain (see below).

## Config

Standard config format with `prior: mcmc` and a `mcmc:` block of sampler knobs.
Shipped emcee configs:
* `data/configs/example_mcmc_config.yaml` — multi-observable demo (full pipeline,
  four constraints).
* `data/configs/example_mcmc_relic_tuned.yaml` — production (full pipeline,
  relic-density Gaussian target, 9 free params, 256 walkers / 128 workers).
* `data/configs/ci_mcmc_config.yaml` — tiny (CI + local smoke test).

Free parameters are those with `lo != hi`; fixed parameters (`lo == hi`) are held
constant and excluded from the sampled dimensions.

Key `mcmc:` keys:

| key | meaning |
| --- | --- |
| `nwalkers` | ensemble size (use `2 × nworkers`; keep even; `≥ 2×ndim`) |
| `nworkers` | pool size / CPU cores → the `chain` rule's cpus-per-task |
| `max_steps` | iteration cap (the wall-clock guard / convergence usually stops earlier) |
| `burn_in` | absolute floor (iterations) on the ntuple burn-in |
| `burn_in_factor` | ntuple burn-in = `max(burn_in, ceil(burn_in_factor × max_tau))` (default 3); kept in sync with `emcee_diagnostics.py` |
| `check_every` | autocorrelation / progress cadence (iterations) |
| `target_autocorr_factor` | require `N > factor × tau` for convergence (default 50) |
| `tau_rel_tol` | and `tau` stable to this relative tolerance (default 0.01) |
| `walltime_hours` | wall-clock guard budget (keep below the `chain` rule's `runtime`) |
| `init.strategy` | `spread` (uniform across prior, broad) or `ball` (around `center_file`) |
| `moves` | emcee moves (default `StretchMove(a=2.0)`; DE-move mix for harder targets) |
| `record_observables` | list of `{name, branch, source}` recorded as blobs → ntuple branches |

`tool_timeout` (top-level, seconds) wraps each external physics-tool call in
`timeout` (threaded through `steps.py`) so a single hung binary can't stall an
ensemble half-step — important for emcee, where `pool.map` blocks on every walker.
**Caveat:** SModelS (and its in-process Pythia cross-section computation) runs
*inside* the Python worker, not as a subprocess, so `tool_timeout` cannot wrap it —
a pathological point that makes SModelS/Pythia hang blocks its worker until the
between-iteration wall-clock guard fires. If you put SModelS in an emcee pipeline,
prefer configs that don't gate the likelihood on it.

## Run locally (smoke test)

Via the power-user Snakemake entry (single mode = one scan dir):

```bash
source build/setup.sh
pixi run smk --workflow-profile ../workflow/profiles/local \
    --config mode=single scan_dir=run/mcmc_smoke \
    config_file=source/Run3ModelGen/data/configs/ci_mcmc_config.yaml seed=0
```

Output lands in `scan_dir`: `chain.h5` (+ `.meta.json`) and `ntuple.0.0.root`
(plus the `chain.json` manifest under `run/r3mg/mcmc_smoke/`). The `genModels.py`
facade (`--config_file … --backend local`) reaches the same rule.

**Resume:** the emcee chain never wipes its scan dir, and `chain.h5` is undeclared
side-state, so once the ntuple exists Snakemake considers the batch *done*. To add
iterations (extend a not-yet-converged chain), force the rule to re-run — it
reopens `chain.h5` and appends:

```bash
pixi run smk --workflow-profile ../workflow/profiles/local --forcerun chain \
    --config mode=single scan_dir=run/mcmc_smoke \
    config_file=source/Run3ModelGen/data/configs/ci_mcmc_config.yaml seed=0
```

(or delete `ntuple.0.0.root` + `chain.json`; `chain.h5` persists and resume kicks
in). Delete the whole scan dir for a fresh chain.

## Run on the cluster (production)

Use the standard facade with `--backend slurm`. Batch mode fans out K independent
ensembles (`scan.<runid>.<b>`), each with its own `blake2b`-derived seed and its
own `chain.h5` — i.e. K independent chains for cross-ensemble R̂:

```bash
genModels.py --config_file source/Run3ModelGen/data/configs/example_mcmc_relic_tuned.yaml \
    --num_jobs 4 --backend slurm
```

The `chain` rule is grouped as one long (23 h) SLURM job per ensemble and requests
`mcmc.nworkers` CPUs. Its memory request defaults to `2000 MB × nworkers`; override
with `--set-resources chain:mem_mb=<n>` (`0` = whole node) for a wide (128-worker)
run. Re-run the same command (or add `--forcerun chain`) to resume ensembles that
haven't converged. **Publishing** (`--publish eos`) *moves* the ROOT ntuple off-node
but leaves `chain.h5` in the local scan dir, so a published ensemble can still be
resumed to add iterations — a re-run regenerates and re-publishes the ntuple. See
`doc/snakemake.md` for the facade/backends and `--publish` details.

## Reproducibility & seeding

`--seed` (batch mode: a `blake2b`-derived per-ensemble seed) seeds the **walker
initialisation** *and* the ensemble sampler's **proposals**, so **an emcee run is
bit-reproducible**: two runs at the same seed give an identical `chain.h5`, identical
ntuple branches and identical per-step counts. It is also invariant to
`mcmc.nworkers` — every proposal is drawn in the parent process and `pool.map`
preserves order, so the worker count changes only how fast the chain is produced, not
what it contains. Verified end-to-end (4 vs 2 workers, bitwise) and pinned by
`tests/test_emcee_determinism.py`.

Two details are worth knowing because they *look* like they should break this:

* Workers are seeded `seed + PID`, so their RNG streams differ from run to run. That
  cannot reach the chain: `evaluate_theta` writes the theta it is handed and runs the
  tools, drawing no randomness (`sampling.py`'s RNG serves the flat/grid/importance
  priors only).
* emcee builds its proposal RNG from `backend.random_state`, falling back to the
  process-wide `np.random.get_state()`. Fresh runs used to reproduce only because
  `ModelGenerator.__init__` calls `np.random.seed(seed)` first — an accident of two
  unrelated implementation details. `seed_sampler_rng()` now assigns the state
  explicitly and **verifies** it took (emcee's setter fails silently), so the guarantee
  no longer depends on that fallback. It is applied only to a fresh chain: a resume
  restores the RNG state saved in the backend, which is what makes a continued leg
  extend the chain rather than retrace it.

This is what lets CI regression-test the chain properly. `prior: mcmc` is
`compare_reference`-gated like every other prior, and additionally on branch **values**
(`compare_values` in `refcheck.SCANS`): `compare_ntuples` checks every branch of
`ntuple.0.0.root` entry by entry against the reference ntuple that
`upload_references.py` publishes. Counts alone would be near-vacuous here — every CI
model clears both steps, so `54/54` holds regardless of where the walkers went — whereas
the values pin the whole trajectory. Because an MCMC chain amplifies numeric differences
rather than averaging them out (a borderline accept/reject that flips diverges the
trajectory), a genuine regression shows up as a wholesale mismatch rather than a subtle
drift.

The comparison uses `refcheck.VALUE_RTOL` (1e-12) rather than bitwise equality: values
do reproduce exactly in practice, but a one-ulp difference in some future SPheno build
would fail an exact gate for no physical reason. Two things the gate depends on, both
worth knowing before changing them:

* **The seed must not move.** `genModels.py --seed` defaults to 123 and neither the CI
  job nor `make_references.py` overrides it. A different default silently invalidates
  every reference.
* **The chain must run to `max_steps`.** The sampling loop's wall-clock guard would
  checkpoint a shorter chain; `compare_ntuples` reports that as a truncation by name
  rather than as a value drift. The CI chain takes ~8 s against a 1 h budget.

Note the guarantee is over a fixed environment: the physics tools are deterministic
binaries, but changing a tool version (or adding a Monte-Carlo step such as SModelS'
pythia cross sections to a chain config) moves the chain, because the likelihood
surface itself moves.

Cross-ensemble R̂/ESS (several independent seeds, `--glob` over multiple `chain.h5`)
is the **authoritative** convergence signal; the within-ensemble figure (walkers as
chains) is a weaker fallback reported when only one ensemble is present.

## Diagnostics

The `chain` rule runs the diagnostics **automatically as its final step**, into
`<scan_dir>/diag/` (a declared directory output): the criteria table + verdict go to
the log and to `diag/diagnostics.txt`, alongside the corner/trace/log-prob and
input-vs-target plots. It is best-effort — a plotting/arviz hiccup never fails an
otherwise-complete chain. Run it by hand for the cross-ensemble view (several seeds)
or to (re)write an ntuple with a chosen discard/thin:

`scripts/emcee_diagnostics.py` reads the HDF5 backend(s) and reports the standard
emcee/arviz diagnostics: mean acceptance fraction, integrated autocorrelation time
τ per parameter, the τ-vs-N convergence curve, trace and log-prob plots, a corner
plot, a scatter+histogram of **each free input dim vs the constrained "target"
observable** (auto-detected — `m_h`, `Omega`, … — or set with `--target`; coloured
by log-prob), and Gelman-Rubin R̂ + bulk/tail ESS via arviz (within each ensemble
and across the independent ensembles). The R̂ block prints a third, **non-gating**
`|M_1|,|M_2| magnitude fold` line: abs()-ing M₁/M₂ is *not* a symmetry (an individual
sign flip moves Ω by ~15–20 %), so it never feeds the verdict — it is a reference
floor, and the gap between it and the signed μ≥0-canonical R̂ is the part of R̂ that
is unmixed sign lobes rather than unconverged magnitudes. It also prints a **burn-in
check** and uses the *same* τ-scaled discard rule as the sampler's ntuple, so the two
stay consistent. It closes with a **criteria table** (every convergence requirement, why
it exists, its reference, and this scan's verdict — see
[Convergence checklist](#convergence-checklist)), opening with a scannable one-line-per-check
SUMMARY, and the CONVERGED / NOT CONVERGED banner. It can emit a ROOT `"susy"` ntuple per
ensemble so `extractModels` (and other ntuple consumers) keep working.

#### Rank statistic (computational faithfulness)

On **all samples pooled** the diagnostics also run a rank-statistic uniformity check
(`rank_all.png`, `rank_ecdf_all.png`): rank every draw of a parameter pooled over every
ensemble and walker, then histogram the ranks *per chain*. If all chains sample the same
distribution each histogram is uniform, and the **shape** of any departure says how they
differ — U-shaped = that chain is over-dispersed relative to the pooled sample,
inverted-U = under-dispersed (concentrated in a sub-region, e.g. trapped in one sign
lobe), skewed = shifted. That shape is the reason to have the check at all: R̂ tells you
*whether* chains disagree, this tells you *how*.

**It is not simulation-based calibration.** Betancourt's workflow §1.2 ranks a known
simulating parameter inside posteriors refitted to simulated data, which certifies that
the posterior is *calibrated*; that needs hundreds of refits, and one fit here is already
~24.5 M full-pipeline evaluations (~12 000 core-hours). This check only certifies that the
chains agree **with each other**. A uniform result is evidence against non-convergence,
never evidence of correct calibration — and note the shape meanings differ from SBC's,
where a U-shape instead means the posterior is too narrow.

Two implementation details that matter if you touch it:

* **Chains are the ensembles, not the walkers.** emcee's DE/snooker moves couple walkers,
  so walker agreement is not evidence of convergence (same reason within-ensemble R̂ is
  only a lower bound). With a single ensemble — which is what every chain job's automatic
  run has — the draws are split into `RANK_SEGMENTS` consecutive time slices instead,
  making it a *stationarity/drift* check; the output labels it as such.
* **The histogram uses every draw; the test does not.** At τ ≈ 2000 the draws are far from
  independent, and a χ² that assumes independence rejects essentially always (measured:
  statistic inflated ≈ 17×, 100 % false-positive rate). Scaling by ESS/N over-corrects by
  an order of magnitude, because bin counts are sums of *indicators* whose autocorrelation
  time is much shorter than the parameter's. So the test thins by τ and uses an exact
  contingency-table homogeneity χ²: pooled ranks fix both margins, giving
  dof = (C−1)(B−1), which is calibrated at its nominal level. A panel can therefore look
  slightly uneven and still pass — read the p-value, not the wiggle. p-values are
  Holm-corrected across parameters, and the row is **advisory**: R̂ already gates
  inter-chain agreement.

```bash
pixi run emcee_diagnostics \
    --glob "/ptmp/$USER/run3modelgen/<runid>/scan.*/chain.h5" \
    --outdir results/<runid>/diag --to-ntuple results/<runid>/ntuples
```

Recommended post-processing (emcee convention): discard ≈ 2–3 × τ_max and thin
≈ τ_max/2 for an independent-sample ntuple.

### Re-running only the diagnostics

The diagnostics run automatically as the chain's final step, but a chain you already
have can be re-diagnosed for free (seconds; `chain.h5` is opened read-only, no
sampling, no pipeline tools):

```bash
pixi run rediagnose run/test_1d          # -> run/test_1d/diag, in place
pixi run rediagnose run/test_1d --to-ntuple ntuples/   # extra args pass through
```

`rediagnose` makes exactly the call the chain rule makes at its end
(`emcee_diagnostics.py --glob <scan_dir>/chain.h5 --outdir <scan_dir>/diag`), so it
reproduces what that run wrote. For the **cross-ensemble** R̂ over several seeds, call
`emcee_diagnostics` directly with a `--glob` spanning the scan dirs, as above — a
single `scan_dir` can only give the within-ensemble (walkers-as-chains) result.

Two things that do *not* work for this:

* `rerunStep.py --step diagnostics` — the diagnostics are the tail of the `chain`
  rule, not a pipeline step, and `rerunStep` refuses `prior: mcmc` outright (it now
  prints the `rediagnose` command instead).
* Deleting `<scan_dir>/diag` and re-running the workflow — `diag` is a *declared*
  output of the `chain` rule, so that re-runs the **sampler**, not just the plots.

Note the mirror image of that last point: because `diag` is a declared directory
output whose *contents* Snakemake does not track, a `rediagnose` that fails halfway
leaves a `diag/` the workflow still considers complete. Check its exit status if you
depend on the report.

## Convergence checklist

The diagnostics print this checklist as a **criteria table** — requirement, why it
exists (with a literature reference), and this scan's value and verdict — directly
above the CONVERGED / NOT CONVERGED banner, so there is no need to reconstruct it
from the numbers above:

| # | criterion | requirement | | reference |
| --- | --- | --- | --- | --- |
| 1 | chain length vs τ | `N > 50 × τ`, every ensemble | hard | [1,2] |
| 2 | Gelman-Rubin R̂ | worst R̂ < 1.05 (< 1.01 stricter) | hard | [3,4] |
| 3 | bulk ESS | min ESS_bulk ≥ 400 | hard | [4] |
| 4 | mean acceptance fraction | 0.2 – 0.5 | advisory | [2,5] |
| 5 | tail ESS | min ESS_tail ≥ 400 | advisory | [4] |
| 6 | burn-in adequacy | discard ≥ log-prob plateau | advisory | [1,2] |
| 7 | rank uniformity | Holm-adjusted p ≥ 0.01 | advisory | [4,6,7] |

Above the rows the table prints a **SUMMARY** line — `hard 1/3 PASS`, `advisory 2/4 ok`,
then one `[verdict] name` cell per check — so "what failed?" is answerable at a glance
without reading the rationale for each.

CONVERGED requires all three **hard** rows. The advisory rows are reported but never
veto the verdict: off-band acceptance flags slow mixing without invalidating a chain
that already meets the R̂/ESS/τ criteria — and in 1–2 free dimensions the stretch
move accepts intrinsically often (its `z^(ndim-1)` factor is ~1), so a high value
there is expected; raise `StretchMove`'s `a` to bring it down. Rank uniformity is
advisory because R̂ already gates inter-chain agreement — its value is the *shape*
diagnosis, not a second veto.

If `N < 50τ`, re-run (with `--forcerun chain`) to add iterations.

The thresholds live in one place — `TAU_FACTOR`, `RHAT_MAX`, `ESS_MIN`, `ACC_BAND`,
`RANK_BINS`, `RANK_P_MIN`, `RANK_SEGMENTS`, `RANK_Z_FLAT` in
`scripts/emcee_diagnostics.py`. The three gating conditions are built once in
`_hard_verdicts()` and read by both the table and the verdict, so they cannot drift
apart.

1. Goodman & Weare 2010, Comm. App. Math. Comp. Sci. **5**, 65 — the affine-invariant
   ensemble sampler (stretch move) emcee implements.
2. Foreman-Mackey, Hogg, Lang & Goodman 2013, PASP **125**, 306 —
   [arXiv:1202.3665](https://arxiv.org/abs/1202.3665) — emcee itself: τ estimation,
   the `N > 50τ` rule of thumb, the acceptance band, the discard/thin convention.
3. Gelman & Rubin 1992, Statist. Sci. **7**, 457 — the original between-chain /
   within-chain variance ratio R̂.
4. Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021, Bayesian Analysis **16**, 667 —
   [arXiv:1903.08008](https://arxiv.org/abs/1903.08008) — rank-normalized split-R̂ and
   bulk/tail ESS as arviz computes them, with the ESS ≥ 400 and R̂ < 1.01 recommendations.
5. Roberts, Gelman & Gilks 1997, Ann. Appl. Probab. **7**, 110 — the optimal-scaling
   result (acceptance → 0.234 in high dimension) behind the healthy band.
6. Talts, Betancourt, Simpson, Vehtari & Gelman 2018 —
   [arXiv:1804.06788](https://arxiv.org/abs/1804.06788) — simulation-based calibration:
   the rank statistic and how to read pathology off its histogram. See also Betancourt's
   [principled Bayesian workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html#12_Computational_Faithfulness)
   §1.2. NB SBC itself is not what the diagnostics compute — see
   [Rank statistic](#rank-statistic-computational-faithfulness).
7. Säilynoja, Bürkner & Vehtari 2022, Stat. Comput. **32**, 32 —
   [arXiv:2103.10522](https://arxiv.org/abs/2103.10522) — the uniform-ECDF difference
   plot with a simultaneous confidence band (`rank_ecdf_all.png`).
