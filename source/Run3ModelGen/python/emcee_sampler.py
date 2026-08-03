'''emcee-based MCMC sampler for pMSSM exploration.

This is a pure orchestration layer on top of the existing physics pipeline
(`Run3ModelGen.modelgen.ModelGenerator`). It IS the MCMC sampler behind
`prior: mcmc`: an affine-invariant ensemble sampler that reuses the pipeline
unchanged via `ModelGenerator.evaluate_theta` (parameter vector -> (chi2,
observables)). It is driven through the normal entry point,
`genModels.py --config_file <mcmc config>` (ModelGenerator.run_emcee).

Key design points (see doc/emcee.md):
  * One `ModelGenerator` is built PER WORKER inside a multiprocessing Pool
    initializer (never pickled), each chdir'd into an isolated working
    directory so parallel `os.system` calls (e.g. superiso's `output.flha`)
    never collide.
  * The log-probability is a module-level function referencing a worker-global
    `ModelGenerator`; it returns (-inf, nan-blobs) for out-of-bounds / invalid
    points and (-chi2/2, observable-blobs) otherwise. It never returns NaN.
  * Full observable vectors are recorded per sample via emcee blobs.
  * Output is an `emcee.backends.HDFBackend` (chain + log_prob + blobs), which
    provides resume-across-jobs and all emcee diagnostics. A converter emits a
    ROOT "susy" ntuple compatible with the existing tooling.
  * The sampling loop monitors the integrated autocorrelation time and stops on
    convergence (N > factor*tau, tau stable) or a wall-clock guard (for the
    24 h SLURM limit), checkpointing every iteration.
'''
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import signal
import time

import numpy as np

import structlog
log = structlog.get_logger()

import emcee

from Run3ModelGen.modelgen import ModelGenerator
from Run3ModelGen.ntupling import input_blocks
from Run3ModelGen.emcee_ntuple import h5_to_ntuple, resolve_burnin  # shared writer + burn-in policy

# Config-parameter -> EXTPAR id(s), mirroring ModelGenerator.run_prep_input.
# Used only by the HDF5 -> ROOT bridge to reproduce the ntuple IN_* branches.
_EXTPAR_IDS = {
    'M_1': [1], 'M_2': [2], 'M_3': [3],
    'AT': [11], 'Ab': [12], 'Atau': [13],
    'mu': [23], 'tanb': [25], 'mA': [26],
    'meL': [31, 32], 'mtauL': [33],
    'meR': [34, 35], 'mtauR': [36],
    'mqL1': [41, 42], 'mqL3': [43],
    'muR': [44, 45], 'mtR': [46],
    'mdR': [47, 48], 'mbR': [49],
}

# ── Worker-global state (set post-fork in init_worker) ──────────────────────
_WORKER: ModelGenerator | None = None
_WORKER_SLOT: int | None = None
_WORKER_RECORD_SOURCES: list = []


def init_worker(config_file: str, base_dir: str, seed: int,
                record_sources: list) -> None:
    '''Pool initializer: build one isolated ModelGenerator for this worker.

    `config_file` and `base_dir` MUST be absolute paths: each worker chdir's into
    its own directory before reading the config. The worker's PID gives a unique,
    respawn-safe working directory (no shared queue to deadlock on).'''
    global _WORKER, _WORKER_SLOT, _WORKER_RECORD_SOURCES
    slot = os.getpid()
    _WORKER_SLOT = slot
    _WORKER_RECORD_SOURCES = record_sources

    wdir = os.path.join(base_dir, f"worker_{slot}")
    scan_dir = os.path.join(wdir, "scan")
    os.makedirs(scan_dir, exist_ok=True)
    # Isolate CWD-relative scratch (superiso's output.flha, CalcHEP temp files).
    os.chdir(wdir)

    # seed + slot keeps workers from sharing an RNG stream, but note it does NOT make
    # the chain PID-dependent: evaluate_theta writes the theta it is handed and runs
    # the tools, drawing no randomness (sampling.py's RNG serves the flat/grid/
    # importance priors only), and every proposal comes from the parent's sampler. So
    # two runs at the same seed agree bitwise despite drawing different PIDs here.
    mg = ModelGenerator(config_file=config_file, scan_dir=scan_dir,
                        seed=seed + slot, quiet=True)
    mg._setup_scan_dirs()
    mg._init_points_arrays(size=1)
    mg._init_fixed_points(modelnum=0)
    _WORKER = mg


def _log_prior(mg: ModelGenerator, theta) -> float:
    '''Flat prior: 0 inside the (possibly interrupted) bounds, -inf outside.'''
    for name, val in zip(mg.free_names, theta):
        scanrange = mg.parameters[name]
        if isinstance(scanrange[0], list):
            if not any(lo <= val <= hi for lo, hi in scanrange):
                return -np.inf
        else:
            lo, hi = scanrange
            if val < lo or val > hi:
                return -np.inf
    return 0.0


def log_prob(theta):
    '''Log-probability + observable blobs for one parameter vector.

    Returns a tuple (logp, obs_1, ..., obs_N). logp is -0.5*chi2 (+flat prior)
    for valid points and -inf for out-of-bounds / pipeline-failed points.
    Never returns NaN for logp; blob length is always len(record_sources).
    '''
    mg = _WORKER
    nrec = len(_WORKER_RECORD_SOURCES)
    nan_blobs = (np.nan,) * nrec

    lp = _log_prior(mg, theta)
    if not np.isfinite(lp):
        return (-np.inf,) + nan_blobs

    try:
        chi2, obs = mg.evaluate_theta(theta, record_sources=_WORKER_RECORD_SOURCES)
    except Exception as exc:  # never let a worker crash the whole ensemble
        log.warning(f"evaluate_theta raised: {exc}")
        return (-np.inf,) + nan_blobs

    # A failed point (bad spectrum, missing step output, non-finite likelihood)
    # comes back as inf from _run_and_evaluate, so the finiteness check vetoes it.
    # A large-but-finite chi2 is a genuine, strongly-disfavoured point and is kept.
    if chi2 is None or not np.isfinite(chi2):
        return (-np.inf,) + nan_blobs

    logp = -0.5 * chi2 + lp
    if not np.isfinite(logp):
        return (-np.inf,) + nan_blobs

    blobs = tuple(obs.get(rec['name'], np.nan) for rec in _WORKER_RECORD_SOURCES)
    return (logp,) + blobs


# ── Walker initialization ───────────────────────────────────────────────────

def _sample_prior_point(parameters: dict, free_names: list, rng: np.random.Generator):
    '''Draw one free-parameter vector uniformly across the prior bounds.
    Interrupted ranges pick a sub-range weighted by its width.'''
    theta = []
    for name in free_names:
        sr = parameters[name]
        if isinstance(sr[0], list):
            widths = np.array([hi - lo for lo, hi in sr], dtype=float)
            idx = rng.choice(len(sr), p=widths / widths.sum())
            theta.append(rng.uniform(sr[idx][0], sr[idx][1]))
        else:
            theta.append(rng.uniform(sr[0], sr[1]))
    return np.array(theta)


def _read_center_file(center_file: str, free_names: list) -> np.ndarray:
    '''Read a seed point (first row) from an importance-sampling-style CSV.'''
    import csv
    with open(center_file) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return np.array([float(row[n]) for n in free_names])


def init_walkers(parameters, free_names, nwalkers, pool, strategy="spread",
                 center_file=None, ball_frac=0.02, oversample=4, max_batches=60,
                 rng=None, deadline=None) -> np.ndarray:
    '''Return an (nwalkers, ndim) array of starting points, all with finite
    log-probability. Rejection-samples through the pool until enough valid
    walkers are collected. If `deadline` (an absolute time.time() value) is given,
    abort with a clear error once it is passed — otherwise a mostly-invalid prior
    can burn the entire wall-clock budget here, before the sampling loop's own
    guard ever runs.'''
    if rng is None:
        # An unseeded default_rng() here used to be the silent path to an
        # unreproducible starting ensemble. There is one caller and it always passes a
        # seeded generator, so refuse rather than quietly draw from OS entropy.
        raise ValueError("init_walkers requires an explicit `rng` "
                         "(np.random.default_rng(seed)); walker initialisation is "
                         "part of the reproducibility contract.")
    ndim = len(free_names)
    if strategy == "ball" and not center_file:
        raise ValueError("init strategy 'ball' requires mcmc.init.center_file "
                         "(the point to scatter the walker ball around).")

    def _propose_batch(n):
        if strategy == "ball":
            center = _read_center_file(center_file, free_names)
            scales = np.array([
                (max(hi - lo for lo, hi in parameters[nm]) if isinstance(parameters[nm][0], list)
                 else parameters[nm][1] - parameters[nm][0])
                for nm in free_names], dtype=float) * ball_frac
            return [center + scales * rng.standard_normal(ndim) for _ in range(n)]
        return [_sample_prior_point(parameters, free_names, rng) for _ in range(n)]

    p0 = []
    for batch_i in range(max_batches):
        batch = _propose_batch(nwalkers * oversample)
        results = pool.map(log_prob, batch)
        for th, res in zip(batch, results):
            lp = res[0] if isinstance(res, (tuple, list)) else res
            if np.isfinite(lp):
                p0.append(th)
        log.info(f"[emcee-init] batch {batch_i + 1}: {len(p0)}/{nwalkers} finite walkers")
        if len(p0) >= nwalkers:
            break
        if deadline is not None and time.time() > deadline:
            raise RuntimeError(
                f"init_walkers: wall-clock deadline hit after {batch_i + 1} batches "
                f"with only {len(p0)}/{nwalkers} finite walkers — the prior is likely "
                f"mostly invalid. Try `ball` init with a center_file, or loosen the "
                f"constraints.")
    if len(p0) < nwalkers:
        raise RuntimeError(
            f"init_walkers: only found {len(p0)}/{nwalkers} finite walkers after "
            f"{max_batches} batches — prior may be mostly invalid. Try `ball` init "
            f"with a center_file, or loosen the constraints.")
    return np.array(p0[:nwalkers])


# ── Moves ───────────────────────────────────────────────────────────────────

def build_moves(moves_cfg):
    '''Translate a config `moves` list into emcee move objects. None -> default
    StretchMove(a=2.0).'''
    if not moves_cfg:
        return None
    out = []
    for m in moves_cfg:
        name = m.get('type')
        if not name or not hasattr(emcee.moves, name):
            valid = sorted(n for n in dir(emcee.moves) if n.endswith('Move'))
            raise ValueError(f"unknown emcee move type {name!r}; valid types: {valid}")
        cls = getattr(emcee.moves, name)
        kwargs = m.get('kwargs', {}) or {}
        weight = float(m.get('weight', 1.0))
        if weight <= 0:
            raise ValueError(f"move {name!r} has non-positive weight {weight}; "
                             f"move weights must be > 0.")
        out.append((cls(**kwargs), weight))
    return out


# ── Reproducibility ─────────────────────────────────────────────────────────

def seed_sampler_rng(sampler, seed: int) -> None:
    '''Pin an EnsembleSampler's proposal RNG to `seed`.

    emcee draws every proposal from `sampler._random`, a RandomState built in its
    __init__ from `backend.random_state` when the backend has one and from the
    process-wide `np.random.get_state()` otherwise. That fallback is why a fresh run
    was *already* bit-reproducible before this function existed: ModelGenerator's
    __init__ calls np.random.seed(seed), and the sampler silently inherited it.
    Reproducible, but only as an accident of two unrelated implementation details --
    it hangs on emcee's undocumented fallback (we allow anything <4), on the
    construction ORDER here, and on nothing else drawing from the global RNG in
    between. Setting the state explicitly makes it a contract instead, one that
    tests/test_emcee_determinism.py pins.

    Call this ONLY on a fresh chain. `sample()` assigns
    `self.random_state = state.random_state` on entry, so a resumed leg is governed by
    the state saved in the backend regardless of what we set here (re-seeding a resume
    is a no-op, not a corruption -- but gate on is_fresh anyway rather than rely on
    that). That same assignment is what lets a fresh seed survive: State(p0) carries
    random_state=None, which emcee's setter rejects silently, leaving our state in
    place.
    '''
    want = np.random.RandomState(seed).get_state()
    sampler.random_state = want          # emcee's setter swallows every exception
    got = sampler.random_state
    if (got[0] != want[0] or not np.array_equal(got[1], want[1])
            or tuple(got[2:]) != tuple(want[2:])):
        raise RuntimeError(
            f"could not pin the ensemble proposal RNG to seed={seed}: emcee's "
            f"random_state setter failed silently (emcee {emcee.__version__}). "
            f"The chain would still run, but would not be reproducible.")


# ── Main driver ─────────────────────────────────────────────────────────────

def run_emcee_scan(config_file, scan_dir, backend_path, seed=0,
                   nwalkers=None, nworkers=None, steps=None, burn_in=None,
                   max_hours=23.0, resume=False, init_strategy=None,
                   center_file=None, to_ntuple=None, worker_base=None):
    '''Run (or resume) an emcee ensemble scan and write an HDF5 backend.

    CLI args override the corresponding keys in the config `emcee:` block.
    Returns a summary dict.
    '''
    # Workers chdir into their own dirs before reading these, so both must be
    # absolute paths.
    config_file = os.path.abspath(config_file)
    scan_dir = os.path.abspath(scan_dir)
    backend_path = os.path.abspath(backend_path)
    # Per-worker scratch (worker_<pid>/ subdirs) lives under worker_base; defaults to
    # scan_dir (legacy standalone behaviour). The Snakemake chain rule points it at the
    # ephemeral job tempdir so scratch stays out of the published/resumable scan_dir.
    worker_base = os.path.abspath(worker_base) if worker_base else scan_dir

    # Read config once in the main process to derive dimensions / knobs.
    master = ModelGenerator(config_file=config_file, scan_dir=scan_dir,
                            seed=seed, quiet=True)
    # Sampler knobs live in the config `mcmc:` block (prior: mcmc == emcee).
    ecfg = getattr(master, 'mcmc', {}) or {}

    free_names = master.free_names
    ndim = len(free_names)
    parameters = master.parameters

    def _pick(cli_val, key, default):
        '''CLI arg wins, then config value (ignoring null), then default.'''
        if cli_val is not None:
            return cli_val
        v = ecfg.get(key)
        return v if v is not None else default

    nwalkers = int(_pick(nwalkers, 'nwalkers', 2 * max(ndim, 2)))
    nworkers = int(_pick(nworkers, 'nworkers', min(8, os.cpu_count() or 1)))
    max_steps = int(_pick(steps, 'max_steps', 100000))
    burn_in = int(_pick(burn_in, 'burn_in', 0))
    # Soft, tau-scaled burn-in for the ntuple: discard = max(burn_in, factor*tau).
    # Default factor 3 (upper end of the emcee 2-3*tau convention): the ntuple is
    # a standalone artifact often analysed without the diagnostics, so its baked-in
    # burn-in errs conservative (under-burning biases; over-burning costs ~1*tau of
    # samples). emcee_diagnostics.py uses the same default, so they stay consistent.
    burn_in_factor = float(ecfg.get('burn_in_factor', 3.0))
    check_every = int(ecfg.get('check_every') or 100)
    target_factor = float(ecfg.get('target_autocorr_factor', 50))
    tau_rel_tol = float(ecfg.get('tau_rel_tol', 0.01))
    init_strategy = init_strategy or ecfg.get('init', {}).get('strategy', 'spread')
    center_file = center_file or ecfg.get('init', {}).get('center_file', None)
    moves = build_moves(ecfg.get('moves', None))

    record_observables = ecfg.get('record_observables', []) or []
    record_sources = [{'name': r['name'], 'source': r['source']} for r in record_observables]
    record_names = [r['name'] for r in record_observables]
    blobs_dtype = [(n, float) for n in record_names] if record_names else None

    min_walkers = 2 * ndim
    if nwalkers < min_walkers:
        log.warning(f"nwalkers={nwalkers} < 2*ndim={min_walkers}; bumping to {min_walkers} "
                    f"(emcee's stretch move requires it).")
        nwalkers = min_walkers
    if nwalkers % 2 != 0:
        nwalkers += 1
        log.warning(f"nwalkers must be even for the stretch move; bumped to {nwalkers}.")

    log.info(f"[emcee] ndim={ndim} free_names={free_names}")
    log.info(f"[emcee] nwalkers={nwalkers} nworkers={nworkers} max_steps={max_steps} "
             f"init={init_strategy} moves={'default' if moves is None else ecfg.get('moves')}")

    os.makedirs(os.path.dirname(os.path.abspath(backend_path)) or '.', exist_ok=True)
    os.makedirs(scan_dir, exist_ok=True)

    # Sidecar metadata for the diagnostics / bridge tooling. It carries the
    # complete ntuple branch layout so the (dependency-light) diagnostics env can
    # convert HDF5 -> ROOT without re-reading the config or importing the package.
    in_branches = _build_in_branch_layout(parameters, free_names)
    try:
        config_sha256 = hashlib.sha256(
            open(config_file, 'rb').read()).hexdigest() if config_file else None
    except OSError:
        config_sha256 = None
    meta = {
        'config_file': os.path.abspath(config_file),
        'config_sha256': config_sha256,
        'free_names': free_names, 'ndim': ndim, 'nwalkers': nwalkers,
        'seed': seed, 'burn_in': burn_in, 'burn_in_factor': burn_in_factor,
        'record_names': record_names,
        'in_branches': in_branches,
        'record_branches': [{'name': r['name'], 'branch': r['branch']}
                            for r in record_observables],
    }

    # Resume guard: extending an existing chain.h5 only makes sense if it was
    # written with the same posterior. A chain resumed under a different config
    # (different free parameters, dimensionality, or any config edit) would
    # silently splice two incompatible ensembles into one .h5 -> garbage samples
    # and, worse, a plausible-looking one. Compare the previous sidecar (before we
    # overwrite it) and refuse the resume on a mismatch.
    meta_path = backend_path + '.meta.json'
    if resume and os.path.exists(backend_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                prev = json.load(f)
        except (OSError, ValueError):
            prev = {}
        mismatched = [k for k in ('free_names', 'ndim', 'config_sha256')
                      if prev.get(k) is not None and prev.get(k) != meta[k]]
        if mismatched:
            raise ValueError(
                f"refusing to resume {backend_path}: it was written with a "
                f"different config (differing {mismatched}). Delete the scan "
                f"directory to start a fresh chain, or restore the original "
                f"config to continue this one.")

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Fork context: children inherit already-imported modules copy-on-write and
    # we build the per-worker ModelGenerator AFTER fork (never pickled).
    ctx = mp.get_context('fork')

    backend = emcee.backends.HDFBackend(backend_path)
    is_fresh = True
    if resume and os.path.exists(backend_path):
        try:
            if backend.iteration > 0:
                is_fresh = False
        except Exception:
            is_fresh = True
    if is_fresh:
        backend.reset(nwalkers, ndim)

    summary = {}
    t_start = time.time()
    # 95% of the wall-clock budget: shared by walker init and the sampling loop so
    # a slow init can't eat the whole allotment before sampling ever starts.
    deadline = t_start + max_hours * 3600.0 * 0.95
    with ctx.Pool(processes=nworkers, initializer=init_worker,
                  initargs=(config_file, worker_base, seed, record_sources)) as pool:

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob, pool=pool, backend=backend,
            moves=moves, blobs_dtype=blobs_dtype)

        if is_fresh:
            # Pin the proposals before the first sample() call. Deliberately NOT on a
            # resume: emcee has already restored the backend's saved RNG state, which
            # is what makes a resumed leg continue the chain instead of repeating it.
            seed_sampler_rng(sampler, seed)
            log.info("[emcee] initializing walkers ...")
            rng = np.random.default_rng(seed)
            initial_state = init_walkers(parameters, free_names, nwalkers, pool,
                                         strategy=init_strategy, center_file=center_file,
                                         rng=rng, deadline=deadline)
        else:
            log.info(f"[emcee] resuming from iteration {backend.iteration}")
            initial_state = backend.get_last_sample()

        remaining = max_steps - backend.iteration
        if remaining <= 0:
            log.info(f"[emcee] already at {backend.iteration} >= max_steps={max_steps}; nothing to do.")
        else:
            # Graceful stop on SLURM pre-kill signals (belt & braces with the
            # wall-clock guard below).
            stop = {'flag': False}

            def _handler(signum, frame):
                stop['flag'] = True
                log.info(f"[emcee] caught signal {signum}; will checkpoint and stop.")
            signal.signal(signal.SIGTERM, _handler)
            try:
                signal.signal(signal.SIGUSR1, _handler)
            except Exception:
                pass

            old_tau = np.inf
            converged = False
            autocorr_trace = []

            for _ in sampler.sample(initial_state, iterations=remaining, store=True, progress=False):
                it = sampler.iteration
                if time.time() > deadline or stop['flag']:
                    log.info(f"[emcee] wall-clock/signal guard at iteration {it}; checkpoint saved.")
                    break
                if it % check_every:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                mean_tau = float(np.nanmean(tau))
                max_tau = float(np.nanmax(tau))
                acc = float(np.mean(sampler.acceptance_fraction))
                autocorr_trace.append([it, mean_tau, max_tau])
                # Convergence needs N > target_factor*tau for every param, so the
                # binding target iteration is target_factor*max_tau. Show it so the
                # progress line makes the remaining burn-in explicit.
                conv_target = target_factor * max_tau
                log.info(f"[emcee-progress] iter={it} mean_tau={mean_tau:.1f} "
                         f"max_tau={max_tau:.1f} target={conv_target:.0f}"
                         f"(={target_factor:.0f}*tau) accept={acc:.3f} "
                         f"elapsed={ (time.time()-t_start)/3600:.2f}h")
                with np.errstate(invalid='ignore', divide='ignore'):
                    stable = np.all(np.abs(old_tau - tau) / tau < tau_rel_tol)
                converged = bool(np.all(tau * target_factor < it) and stable)
                old_tau = tau
                if converged:
                    log.info(f"[emcee] CONVERGED at iteration {it} "
                             f"(N > {target_factor:.0f}*tau and dtau/tau < {tau_rel_tol}).")
                    break

            summary['converged'] = converged
            summary['autocorr_trace'] = autocorr_trace

        # Final acceptance / autocorr summary from the full backend. acceptance_fraction
        # is empty if no iterations ever ran (fresh chain that hit the deadline in init,
        # or an already-complete resume) -> report nan rather than a RuntimeWarning.
        try:
            mean_acc = float(np.mean(sampler.acceptance_fraction))
        except Exception:
            mean_acc = float('nan')
        try:
            tau_final = sampler.get_autocorr_time(tol=0)
            max_tau_final = float(np.nanmax(tau_final))
        except Exception:
            max_tau_final = float('nan')

    n_iter = int(backend.iteration)
    n_samples = n_iter * nwalkers
    summary.update({
        'iterations': n_iter, 'nwalkers': nwalkers, 'n_samples': n_samples,
        'mean_acceptance': mean_acc, 'max_tau': max_tau_final,
        'wall_hours': (time.time() - t_start) / 3600.0,
        'backend': os.path.abspath(backend_path),
    })

    log.info(f"[emcee-summary] iterations={n_iter} nwalkers={nwalkers} "
             f"samples={n_samples} mean_accept={mean_acc:.3f} max_tau={max_tau_final:.1f} "
             f"n50tau={'yes' if (n_iter > 50 * max_tau_final) else 'no'} "
             f"wall_h={summary['wall_hours']:.2f}")

    if to_ntuple and n_iter == 0:
        log.warning(f"[emcee] no iterations in {backend_path}; skipping ntuple write "
                    f"(nothing to convert).")
    elif to_ntuple:
        discard = resolve_burnin(max_tau_final, n_iter, burn_in, burn_in_factor)
        log.info(f"[emcee] writing ntuple -> {to_ntuple} (burn-in discard={discard} "
                 f"= max(burn_in={burn_in}, ceil({burn_in_factor:g}*max_tau={max_tau_final:.1f})))")
        n_written, n_branches = h5_to_ntuple(backend_path, to_ntuple, discard=discard, thin=1)
        log.info(f"[emcee] wrote {n_written} samples to {to_ntuple} ({n_branches} branches)")

    return summary


# ── HDF5 -> ROOT "susy" ntuple bridge ───────────────────────────────────────

def _build_in_branch_layout(parameters: dict, free_names: list) -> list:
    '''Precompute the IN_* branch layout (mirrors run_prep_input's EXTPAR map)
    so the converter can build the ntuple from chain columns / fixed constants
    without re-deriving the mapping. Returns a list of dicts:
      {branch, kind: 'free', index} or {branch, kind: 'fixed', value}.'''
    layout = []
    for pname, ids in _EXTPAR_IDS.items():
        if pname not in parameters:
            continue
        for extid in ids:
            branch = "IN_" + input_blocks['EXTPAR'][extid]
            if pname in free_names:
                layout.append({'branch': branch, 'kind': 'free',
                               'index': free_names.index(pname)})
            else:
                sr = parameters[pname]
                const = sr[0][0] if isinstance(sr[0], list) else sr[0]
                layout.append({'branch': branch, 'kind': 'fixed', 'value': float(const)})
    return layout
