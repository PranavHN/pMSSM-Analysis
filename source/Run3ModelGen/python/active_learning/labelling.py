'''Label a batch of points with the physics pipeline, into an iteration scan dir.

Torch-free. The parent process never runs a tool: a ``multiprocessing`` pool of
``nworkers`` processes (``forkserver`` context, see :func:`start_pool`) each hold one
:class:`~Run3ModelGen.modelgen.ModelGenerator` and run the gated step chain for one model
index at a time, exactly as ``ModelGenerator.generate_models`` does for a flat scan — a
step is attempted only if the step producing its ``input_dir`` succeeded, and per-model
tool failure is data, not an error. The iteration dir therefore looks like any other scan
dir (``input/``, ``SPheno/``, ``micromegas/`` ..., ``scan_config.yaml``,
``ntuple.0.0.root``), with the model index continuing across the labelling attempts of
one iteration so that one ntuple holds every attempt.

Why forkserver: the loop in the parent holds a HIP/CUDA context once the surrogate has
trained, and a plain ``fork`` of such a process is undefined behaviour. The forkserver is
a fresh interpreter (``exec``), so the workers never see the GPU runtime.
'''
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import numpy as np

import structlog
log = structlog.get_logger()

from Run3ModelGen import pipeline
from Run3ModelGen.modelgen import ModelGenerator

# ── worker side ────────────────────────────────────────────────────────────────

_WORKER: dict = {}


def _init_worker(config_file: str, scratch_base: str, seed: int) -> None:
    '''Pool initializer: one ModelGenerator per worker, cwd in a private scratch dir
    (CalcHEP/SPheno drop files into the cwd). ``scan_dir`` is rebound per task.'''
    # Tools and BLAS: one thread per worker, as r3mg_batch.py pins for the emcee pool.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    wdir = os.path.join(scratch_base, f"worker_{os.getpid()}")
    os.makedirs(wdir, exist_ok=True)
    os.chdir(wdir)
    mg = ModelGenerator(config_file=config_file or None, scan_dir=os.path.join(wdir, "unbound"),
                        seed=seed, quiet=True)
    # Filenames are indexed by model number, so the point arrays must span the largest
    # index an iteration dir can hold.
    mg._init_points_arrays(size=pipeline.MAX_MODELS_PER_JOB)
    _WORKER["mg"] = mg
    _WORKER["producer_of"] = {s["output_dir"]: s["name"] for s in mg.steps if s.get("output_dir")}


def _label_one(task) -> tuple:
    '''Run the gated step chain for one model: ``(iteration_dir, index, point)`` ->
    ``(index, {step: tool_ran}, {step: produced_ntuple_vars})``. Never raises.'''
    iteration_dir, idx, point = task
    mg = _WORKER["mg"]
    producer_of = _WORKER["producer_of"]
    mg.scan_dir = iteration_dir
    mg._setup_scan_dirs()
    for key, value in point.items():
        mg.points[key][idx] = value

    ran, usable = {}, {}
    for step in mg.steps:
        indir = step.get("input_dir")
        gate = producer_of.get(indir) if indir else None
        if gate is not None and not ran.get(gate, False):
            continue
        try:
            ok = bool(mg.run_step(modelnum=idx, **step))
        except Exception as exc:   # a tool wrapper raising is a failed model, not a dead worker
            log.warning(f"model {idx}: step {step['name']} raised {exc!r}; counted as failed")
            ok = False
        ran[step["name"]] = ok
        if ok:
            try:
                usable[step["name"]] = bool(mg.step_produced_ntuple_vars(idx, step))
            except Exception:
                usable[step["name"]] = False
    return idx, ran, usable


def start_pool(nworkers: int, config_file: str | None, scratch_base: str, seed: int):
    '''The labelling pool. Create it ONCE per run, before the surrogate touches the GPU.'''
    os.makedirs(scratch_base, exist_ok=True)
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["Run3ModelGen.modelgen"])
    return ctx.Pool(processes=int(nworkers), initializer=_init_worker,
                    initargs=(os.path.abspath(config_file) if config_file else None,
                              os.path.abspath(scratch_base), int(seed)))


# ── parent side ────────────────────────────────────────────────────────────────

STEPS_JSON = "steps.json"


@dataclass
class LabelledBatch:
    '''What one labelling attempt produced, in the order of the submitted points.'''
    indices: np.ndarray      # model indices inside the iteration dir
    y: np.ndarray            # raw target branch (-1 when the pipeline produced nothing)
    valid: np.ndarray        # target window + every veto
    hit: np.ndarray          # within the hit band
    counts: dict             # {step: [with_ntuple_vars, tool_ran, attempted]}
    seconds: float


def _write_steps_json(iteration_dir: str, attempt: int, first_index: int, steps: list,
                      results: list, seconds: float) -> dict:
    '''Append this attempt's per-step tallies to ``<iteration_dir>/steps.json`` (the
    manifest schema of ``pipeline.write_manifest``, one block per attempt).'''
    path = os.path.join(iteration_dir, STEPS_JSON)
    doc = {"attempts": [], "num_models": 0}
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    n = len(results)
    indices = sorted(r[0] for r in results)
    block = {"attempt": int(attempt), "first_index": int(first_index), "num_models": n,
             "seconds": round(float(seconds), 3), "steps": {}}
    counts = {}
    for step in steps:
        name = step["name"]
        attempted = [r[0] for r in results if name in r[1]]
        succeeded = [r[0] for r in results if r[1].get(name)]
        usable = [r[0] for r in results if r[2].get(name)]
        block["steps"][name] = {
            "attempted": sorted(attempted), "succeeded": sorted(succeeded),
            "with_ntuple_vars": sorted(usable),
            "counts": [len(usable), n], "counts_tool_ran": [len(succeeded), n],
        }
        counts[name] = [len(usable), len(succeeded), len(attempted)]
    block["scheduler"] = pipeline.scheduler_identity()
    doc["attempts"].append(block)
    doc["num_models"] = max(doc.get("num_models", 0), (max(indices) + 1) if indices else 0)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)
    return counts


def read_labels(ntuple_path: str, target, vetoes, indices) -> tuple:
    '''``(y, valid, hit)`` for the given model indices of an iteration ntuple.

    The branch is read by name; a branch missing from the ntuple (no model of the
    iteration produced it) yields the ``-1`` sentinel for every row, i.e. invalid.'''
    import uproot

    indices = np.asarray(indices, dtype=int)
    with uproot.open(ntuple_path) as f:
        tree = f["susy"]
        n = int(tree.num_entries)
        available = set(tree.keys())
        wanted = [target.branch] + [v.branch for v in vetoes]
        present = [b for b in dict.fromkeys(wanted) if b in available]
        arrays = tree.arrays(present, library="np") if present else {}
    if indices.size and indices.max() >= n:
        raise RuntimeError(f"{ntuple_path} has {n} entries but labels for index "
                           f"{int(indices.max())} were requested: the iteration ntuple "
                           f"is out of sync with its directory")

    def column(branch):
        if branch in arrays:
            return np.asarray(arrays[branch], dtype=float)[indices]
        log.warning(f"branch {branch!r} is absent from {ntuple_path}: no model produced it")
        return np.full(len(indices), -1.0)

    y = column(target.branch)
    valid = target.is_valid(y)
    for veto in vetoes:
        valid &= veto.passes(column(veto.branch))
    hit = valid & target.is_hit(y)
    return y, valid, hit


def label_batch(pool, config_file: str | None, iteration_dir: str, points: dict,
                first_index: int, attempt: int, seed: int, target, vetoes,
                logger=None) -> LabelledBatch:
    '''Label ``points`` (``{param: array}``, all 19 parameters) as models
    ``first_index .. first_index + n - 1`` of ``iteration_dir``, rebuild the iteration's
    ntuple over every model labelled so far, and read the labels back.'''
    logger = logger or log
    iteration_dir = os.path.abspath(iteration_dir)
    n = len(next(iter(points.values())))
    if first_index + n > pipeline.MAX_MODELS_PER_JOB:
        raise RuntimeError(f"iteration dir {iteration_dir} would hold {first_index + n} models, "
                           f"above the {pipeline.MAX_MODELS_PER_JOB} per-directory cap")
    indices = np.arange(first_index, first_index + n)
    tasks = [(iteration_dir, int(i), {k: float(v[j]) for k, v in points.items()})
             for j, i in enumerate(indices)]

    # The parent's own generator: owns the iteration dir layout, the config dump and the
    # ntuple; it never runs a tool. quiet=True: the banner is printed once per rule.
    mg = ModelGenerator(config_file=config_file or None, scan_dir=iteration_dir,
                        seed=seed, quiet=True)
    mg._setup_scan_dirs()

    logger.info(f"labelling {n} models [{first_index}, {first_index + n}) of {iteration_dir} "
                f"(attempt {attempt}) on {pool._processes} workers")
    t0 = time.time()
    results = []
    for k, res in enumerate(pool.imap(_label_one, tasks, chunksize=1), 1):
        results.append(res)
        if k % 100 == 0 or k == n:
            logger.info(f"\t{k}/{n} models done ({time.time() - t0:.0f} s)")
    seconds = time.time() - t0

    counts = _write_steps_json(iteration_dir, attempt, first_index, mg.steps, results, seconds)
    summary = ", ".join(f"{s}: {c[0]}/{c[1]}/{c[2]}" for s, c in counts.items())
    logger.info(f"attempt {attempt} per-step counts (with ntuple vars / tool ran / attempted): {summary}")

    # The ntuple spans every model of the iteration (all attempts so far); the iteration's
    # scan_config.yaml records the merged config, scan dir and seed as provenance.
    mg.num_models = first_index + n
    mg.mkntuple()
    mg._dump_scan_config()

    y, valid, hit = read_labels(pipeline.ntuple_name(iteration_dir), target, vetoes, indices)
    logger.info(f"attempt {attempt}: {int(valid.sum())}/{n} valid, {int(hit.sum())}/{n} hits "
                f"(|{target.branch} - {target.value:g}| / {abs(target.value):g} < {target.hit_tolerance:g}), "
                f"{seconds:.0f} s")
    return LabelledBatch(indices=indices, y=y, valid=valid, hit=hit, counts=counts, seconds=seconds)
