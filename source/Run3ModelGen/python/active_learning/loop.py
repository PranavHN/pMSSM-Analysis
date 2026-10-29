'''The active-learning loop: ``run_active_learning_scan`` is the body of the
``active_learning`` Snakemake rule (via ``r3mg_batch.py active_learning``).

    iteration_000   seed_models uniform draws from `parameters`, labelled by the pipeline
    iteration_NNN   train the surrogate on every valid labelled model (80/20 train/val);
                    draw n_candidates (LHS/uniform); score; select num_models; label them;
                    retry with a fresh pool until min_valid_fraction * num_models are valid
                    (max_attempts), every attempt appending to the same iteration dir;
                    append the valid models to the labelled set; save al_state.pt

Resume: ``al_state.pt`` (saved atomically after every iteration) plus the iteration dirs
are undeclared side-state of the rule, so re-running it continues after the last
finished iteration; an iteration dir without its ``al_iteration.json`` was interrupted and
is redone. The wall-clock guard and SIGTERM stop the loop BETWEEN iterations with
``complete=False``; the rule body then exits non-zero without the scan's ntuple, so
Snakemake re-runs (retries) or the user re-runs the same command.

GPU: the loop refuses to run without one (``require_gpu``); the labelling pool never
touches it (forkserver workers, started before the surrogate exists).
'''
from __future__ import annotations

import csv
import json
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime

import numpy as np

import structlog
log = structlog.get_logger()

from Run3ModelGen import pipeline
# Imported HERE, before any log handler is attached: modelgen's import calls
# structlog.stdlib.recreate_defaults(), which re-creates the root logging config
# (basicConfig(force=True)) and would drop a handler attached earlier.
from Run3ModelGen import modelgen as _modelgen  # noqa: F401
from Run3ModelGen.active_learning import data as aldata
from Run3ModelGen.active_learning.config import AlConfig, load_al_config
from Run3ModelGen.active_learning.labelling import label_batch, start_pool
from Run3ModelGen.active_learning.ntuple import SELECTION_JSON, merge_iterations

STATE_FILE = "al_state.pt"
MAIN_LOG = "active_learning.log"
ITERATION_JSON = "al_iteration.json"
SUMMARY_JSON = "al_summary.json"
STOP_EXIT_CODE = 3     # "checkpointed, re-run to resume" (distinct from an error)


class StopRequested(Exception):
    '''The wall-clock guard or a SIGTERM asked the loop to checkpoint and stop.'''


def require_gpu():
    '''FATAL unless torch is importable and sees a GPU: prior: active_learning has no
    CPU mode by design. Returns the torch device.'''
    try:
        import torch
    except ImportError as exc:
        log.fatal("FATAL: prior: active_learning needs PyTorch, which is not installed in this "
                  "environment. Install and use a GPU environment: pixi install -e rocm (or cuda) "
                  f"and run from it (pixi run -e rocm ...). [{exc}]")
        sys.exit(1)
    if not torch.cuda.is_available():
        log.fatal("FATAL: no GPU is visible to PyTorch (torch.cuda.is_available() is False). "
                  "prior: active_learning has no CPU mode: run it on a GPU node "
                  "(--backend slurm, whose profile requests a GPU for the active_learning "
                  "rule, or a GPU machine for --backend local).")
        sys.exit(1)
    return torch.device("cuda")


def iteration_dir(scan_dir: str, it: int) -> str:
    return os.path.join(scan_dir, f"iteration_{it:03d}")


def _attach_main_log(scan_dir: str):
    '''Append the structlog console output to <scan_dir>/active_learning.log as well.'''
    handler = logging.FileHandler(os.path.join(scan_dir, MAIN_LOG))
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return handler


def _json_dump(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── state ──────────────────────────────────────────────────────────────────────

def save_state(scan_dir: str, labelled: aldata.LabelledSet, iteration: int, history: list,
               warm: dict | None, rng_state: dict) -> None:
    import torch
    path = os.path.join(scan_dir, STATE_FILE)
    tmp = path + ".tmp"
    torch.save({"iteration": int(iteration), "labelled": labelled.to_dict(), "history": history,
                "warm": warm, "rng": rng_state, "saved": datetime.now().isoformat(timespec="seconds")},
               tmp)
    os.replace(tmp, path)


def load_state(scan_dir: str):
    import torch
    path = os.path.join(scan_dir, STATE_FILE)
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=False, map_location="cpu")


def capture_rng() -> dict:
    import torch
    return {"torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(rng: dict) -> None:
    import torch
    torch.random.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:
            pass
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


# ── per-iteration files ────────────────────────────────────────────────────────

def _append_selected_csv(it_dir: str, free_names: list, points: dict, pred, score, attempt: int,
                         first_index: int) -> None:
    '''The benchmark's selected_points.csv: one row per labelled point, all parameters,
    plus the prediction/score and the model index.'''
    path = os.path.join(it_dir, "selected_points.csv")
    new = not os.path.exists(path)
    names = list(points.keys())
    n = len(points[names[0]])
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["index", "attempt"] + names + ["pred", "score"])
        for j in range(n):
            w.writerow([first_index + j, attempt] + [f"{points[k][j]:.10g}" for k in names]
                       + [f"{pred[j]:.6g}" if np.isfinite(pred[j]) else "",
                          f"{score[j]:.6g}" if np.isfinite(score[j]) else ""])


def _append_selection_json(it_dir: str, strategy: str, attempt: int, first_index: int,
                           pred, score, batch, pool_info: dict) -> None:
    path = os.path.join(it_dir, SELECTION_JSON)
    doc = {"strategy": strategy, "attempts": [], "models": []}
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    doc["attempts"].append({"attempt": attempt, "first_index": first_index, "n": len(pred),
                            **pool_info})
    for j in range(len(pred)):
        doc["models"].append({"index": first_index + j, "attempt": attempt,
                              "pred": None if not np.isfinite(pred[j]) else float(pred[j]),
                              "score": None if not np.isfinite(score[j]) else float(score[j]),
                              "valid": bool(batch.valid[j]), "hit": bool(batch.hit[j])})
    _json_dump(path, doc)


# ── the loop ───────────────────────────────────────────────────────────────────

def run_active_learning_scan(config_file, scan_dir, seed, nworkers=None, max_hours=None,
                             worker_base=None, to_ntuple=None, device=None, logger=None,
                             batch_index=0) -> dict:
    '''Run (or resume) an active-learning scan; returns the summary dict (``complete`` says
    whether every iteration is done). ``device`` defaults to the GPU (FATAL without one).'''
    import torch

    logger = logger or log
    config_file = os.path.abspath(config_file) if config_file else None
    scan_dir = os.path.abspath(scan_dir)
    os.makedirs(scan_dir, exist_ok=True)
    worker_base = os.path.abspath(worker_base) if worker_base else os.path.join(scan_dir, ".workers")
    to_ntuple = to_ntuple or pipeline.ntuple_name(scan_dir)

    merged = pipeline.load_scan_config(config_file, f"{pipeline_datadir()}/configs/default_config.yaml")
    cfg: AlConfig = load_al_config(merged)
    parameters = merged["parameters"]
    free_names = aldata.free_parameters(parameters)
    lo, hi = aldata.free_box(parameters, free_names)
    nworkers = int(nworkers or cfg.nworkers)
    max_hours = float(max_hours or cfg.walltime_hours)
    device = torch.device(device) if device is not None else require_gpu()

    main_handler = _attach_main_log(scan_dir)
    t_start = time.time()
    deadline = t_start + max_hours * 3600.0 * 0.95
    stop = {"flag": False}

    def _handler(signum, frame):
        stop["flag"] = True
        logger.info(f"[active_learning] caught signal {signum}; stopping after the current step")
    signal.signal(signal.SIGTERM, _handler)
    try:
        signal.signal(signal.SIGUSR1, _handler)
    except Exception:
        pass

    logger.info(f"[active_learning] model={cfg.model} head={cfg.head} strategy={cfg.strategy} "
                f"target={cfg.target.branch} (value {cfg.target.value:g}, {cfg.target.transform}) "
                f"free parameters={free_names} seed_models={cfg.seed_models} "
                f"num_models={cfg.num_models}/iteration n_iterations={cfg.n_iterations} "
                f"nworkers={nworkers} device={device}")

    # The labelling pool: one per run, before torch touches the GPU (forkserver, so it
    # would be safe later too; see labelling.py).
    pool = start_pool(nworkers, config_file, worker_base, seed)
    try:
        return _run(cfg, merged, parameters, free_names, lo, hi, config_file, scan_dir, seed, pool,
                    device, deadline, stop, to_ntuple, logger, batch_index, t_start)
    finally:
        pool.close()
        pool.join()
        logging.getLogger().removeHandler(main_handler)
        main_handler.close()


def pipeline_datadir() -> str:
    return os.environ["DATAPATH"].split(":")[0]


def _iteration_done(it_dir: str) -> bool:
    return os.path.exists(os.path.join(it_dir, ITERATION_JSON))


def _fresh_iteration_dir(it_dir: str, logger) -> None:
    '''Remove an interrupted iteration dir (one without al_iteration.json).'''
    import shutil
    if os.path.isdir(it_dir):
        logger.info(f"removing interrupted iteration dir {it_dir}")
        shutil.rmtree(it_dir)
    os.makedirs(it_dir, exist_ok=True)


def _check_stop(stop, deadline, last_iteration_seconds, logger, where: str) -> None:
    if stop["flag"]:
        raise StopRequested(f"signal received ({where})")
    left = deadline - time.time()
    if left <= 0 or (last_iteration_seconds and left < 1.2 * last_iteration_seconds):
        raise StopRequested(f"wall-clock guard: {left/3600:.2f} h left, last iteration took "
                            f"{(last_iteration_seconds or 0)/3600:.2f} h ({where})")


def _run(cfg, merged, parameters, free_names, lo, hi, config_file, scan_dir, seed, pool, device,
         deadline, stop, to_ntuple, logger, batch_index, t_start) -> dict:
    import torch
    from Run3ModelGen.active_learning.acquisition import select
    from Run3ModelGen.active_learning.models import make_surrogate

    target, vetoes = cfg.target, cfg.vetoes
    state = load_state(scan_dir)
    if state is not None:
        labelled = aldata.LabelledSet.from_dict(state["labelled"])
        start_iteration = int(state["iteration"]) + 1
        history = list(state["history"])
        warm = state.get("warm")
        restore_rng(state["rng"])
        logger.info(f"[active_learning] resuming after iteration {state['iteration']} "
                    f"({len(labelled)} labelled models, saved {state.get('saved')})")
    else:
        labelled = aldata.LabelledSet(free_names=free_names)
        start_iteration, history, warm = 0, [], None
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        random.seed(int(seed))

    summary = {"model": cfg.model, "head": cfg.head, "strategy": cfg.strategy, "seed": int(seed),
               "n_iterations": cfg.n_iterations, "target": target.branch, "scan_dir": scan_dir,
               "complete": False, "iterations": history, "iterations_done": start_iteration - 1}
    last_seconds = history[-1]["seconds"] if history else None
    rng_split = np.random.default_rng(int(seed) * 7 + 1)   # train/val roles

    try:
        for it in range(start_iteration, cfg.n_iterations + 1):
            _check_stop(stop, deadline, last_seconds, logger, f"before iteration {it}")
            it_dir = iteration_dir(scan_dir, it)
            if _iteration_done(it_dir) and it < start_iteration:
                continue
            _fresh_iteration_dir(it_dir, logger)
            t_it = time.time()
            rec = {"iteration": it, "attempts": [], "n_labelled": 0, "n_valid": 0, "n_hits": 0,
                   "n_before": len(labelled)}
            torch.manual_seed(int(seed) * 7919 + it)

            if it == 0:
                # ── the seed set: a flat scan of seed_models points ──
                logger.info(f"[iteration 000] labelling {cfg.seed_models} uniform seed models")
                X = aldata.generate_candidates(parameters, cfg.seed_models, "uniform", int(seed), free_names)
                points = aldata.to_points(X, parameters, free_names)
                batch = label_batch(pool, config_file, it_dir, points, 0, 0, seed, target, vetoes, logger)
                role = aldata.split_roles(len(X), cfg.val_fraction, rng_split)
                labelled.append(X, batch.y, batch.valid, batch.hit, role, 0, 0, batch.indices)
                pred = score = np.full(len(X), np.nan)
                _append_selected_csv(it_dir, free_names, points, pred, score, 0, 0)
                _append_selection_json(it_dir, "seed", 0, 0, pred, score, batch, {"generation": "uniform"})
                rec["attempts"].append({"attempt": 0, "n": len(X), "n_valid": int(batch.valid.sum()),
                                        "n_hits": int(batch.hit.sum()), "seconds": batch.seconds,
                                        "counts": batch.counts})
                rec["n_labelled"] = len(X)
                rec["n_valid"] = int(batch.valid.sum())
                rec["n_hits"] = int(batch.hit.sum())
            else:
                # ── train ──
                tr, va = labelled.train_rows(), labelled.val_rows()
                if len(tr) < 2 or len(va) < 1:
                    raise RuntimeError(f"iteration {it}: only {len(tr)} training / {len(va)} validation "
                                       f"models are valid; the seed set is too small or every model failed")
                t_all = target.transform_y(labelled.y)
                surrogate = make_surrogate(cfg, len(free_names), device, lo, hi, int(seed) * 7919 + it)
                logger.info(f"[iteration {it:03d}] training {cfg.model} ({cfg.head}) on {len(tr)} train / "
                            f"{len(va)} val models{' (warm start)' if (warm and cfg.training['warm_start']) else ''}")

                def _progress(epoch, trl, val, _it=it):
                    logger.info(f"[iteration {_it:03d}] epoch {epoch} train {trl:.5f} val {val:.5f}")

                metrics = surrogate.fit(labelled.X[tr], t_all[tr], labelled.X[va], t_all[va],
                                        warm=warm if cfg.training["warm_start"] else None,
                                        log_path=os.path.join(it_dir, "training.log"),
                                        progress=_progress)
                ckpt = surrogate.checkpoint()
                torch.save(ckpt, os.path.join(it_dir, "model.pt"))
                _json_dump(os.path.join(it_dir, "metrics.json"), metrics)
                rec["training"] = {k: v for k, v in metrics.items() if k not in ("train_losses", "val_losses")}
                logger.info(f"[iteration {it:03d}] trained: best epoch {metrics['best_epoch']}, "
                            f"best val {metrics['loss_name']} {metrics['best_val_loss']:.5f}, "
                            f"verdict accuracy val {metrics['verdict_accuracy_val']:.4f}, "
                            f"{metrics['seconds']:.0f} s")
                _check_stop(stop, deadline, None, logger, f"after training of iteration {it}")

                # ── select + label, with retries ──
                n_target = max(1, int(cfg.num_models * cfg.labelling["min_valid_fraction"]))
                first_index, n_collected = 0, 0
                new_X, new_rows = [], []
                for attempt in range(cfg.labelling["max_attempts"]):
                    if first_index + cfg.num_models > pipeline.MAX_MODELS_PER_JOB:
                        logger.warning(f"[iteration {it:03d}] attempt {attempt} would exceed the "
                                       f"{pipeline.MAX_MODELS_PER_JOB}-model cap of one iteration dir; stopping attempts")
                        break
                    cand_seed = int(seed) * 10_000 + it if attempt == 0 else int(seed) * 10_000 + it * 1000 + attempt
                    sel_seed = int(seed) * 100_000 + it if attempt == 0 else cand_seed
                    X_cand = aldata.generate_candidates(parameters, cfg.n_candidates, cfg.candidate_generation,
                                                        cand_seed, free_names)
                    logger.info(f"[iteration {it:03d}] attempt {attempt}: scoring {len(X_cand)} "
                                f"{cfg.candidate_generation} candidates")
                    ev = surrogate.evaluate(X_cand)
                    sel = select(cfg.strategy, surrogate, ev, cfg.num_models, cfg.acquisition,
                                 cfg.threshold, sel_seed, device, logger)
                    X_sel = X_cand[sel.indices]
                    dup = labelled.is_duplicate(X_sel) if len(labelled) else np.zeros(len(X_sel), bool)
                    if dup.any():
                        logger.info(f"[iteration {it:03d}] {int(dup.sum())} selected points duplicate labelled ones; dropped")
                        X_sel, pred, score = X_sel[~dup], sel.pred[~dup], sel.score[~dup]
                    else:
                        pred, score = sel.pred, sel.score
                    if len(X_sel) == 0:
                        continue
                    points = aldata.to_points(X_sel, parameters, free_names)
                    batch = label_batch(pool, config_file, it_dir, points, first_index, attempt, seed,
                                        target, vetoes, logger)
                    _append_selected_csv(it_dir, free_names, points, pred, score, attempt, first_index)
                    pool_info = {"n_candidates": int(len(X_cand)), "candidate_seed": cand_seed,
                                 "pred_mean_range": [float(ev.mean.min()), float(ev.mean.max())],
                                 "score_range": [float(np.nanmin(ev.summary["score"].numpy())),
                                                 float(np.nanmax(ev.summary["score"].numpy()))]}
                    _append_selection_json(it_dir, cfg.strategy, attempt, first_index, pred, score, batch, pool_info)
                    role = aldata.split_roles(len(X_sel), cfg.val_fraction, rng_split)
                    labelled.append(X_sel, batch.y, batch.valid, batch.hit, role, it, attempt,
                                    batch.indices, pred, score)
                    rec["attempts"].append({"attempt": attempt, "n": int(len(X_sel)),
                                            "n_valid": int(batch.valid.sum()), "n_hits": int(batch.hit.sum()),
                                            "seconds": batch.seconds, "counts": batch.counts})
                    rec["n_labelled"] += int(len(X_sel))
                    rec["n_valid"] += int(batch.valid.sum())
                    rec["n_hits"] += int(batch.hit.sum())
                    first_index += len(X_sel)
                    n_collected += int(batch.valid.sum())
                    if n_collected >= n_target:
                        break
                    if attempt < cfg.labelling["max_attempts"] - 1:
                        logger.info(f"[iteration {it:03d}] {n_collected}/{n_target} valid models so far; "
                                    f"retrying with a fresh candidate pool")
                    if stop["flag"]:
                        break
                warm = ckpt

            rec["seconds"] = time.time() - t_it
            rec["n_after"] = len(labelled)
            rec["n_train_valid"] = int(len(labelled.train_rows()))
            rec["n_val_valid"] = int(len(labelled.val_rows()))
            history.append(rec)
            _json_dump(os.path.join(it_dir, ITERATION_JSON), rec)
            save_state(scan_dir, labelled, it, history, warm, capture_rng())
            summary["iterations_done"] = it
            last_seconds = rec["seconds"]
            logger.info(f"[iteration {it:03d}] done: +{rec['n_labelled']} labelled, {rec['n_valid']} valid, "
                        f"{rec['n_hits']} hits; labelled set {len(labelled)} "
                        f"({rec['n_train_valid']} train / {rec['n_val_valid']} val valid); "
                        f"{rec['seconds']:.0f} s; {(time.time() - t_start)/3600:.2f} h elapsed")
            if stop["flag"]:
                raise StopRequested("signal received (after an iteration)")
    except StopRequested as why:
        logger.warning(f"[active_learning] stopped: {why}. State is checkpointed after iteration "
                       f"{summary['iterations_done']}; re-run the same command to resume.")
        summary["stopped"] = str(why)
        summary["complete"] = False
        return summary

    # ── done: the scan's ntuple and the summary ──
    n_models, n_branches = merge_iterations(scan_dir, to_ntuple, target, vetoes, batch_index, logger)
    summary.update({"complete": True, "n_models": n_models, "n_branches": n_branches,
                    "n_valid": int(labelled.valid.sum()), "n_hits": int(labelled.hit.sum()),
                    "wall_hours": (time.time() - t_start) / 3600.0})
    _json_dump(os.path.join(scan_dir, SUMMARY_JSON), summary)
    logger.info(f"[active_learning] complete: {n_models} models in {to_ntuple} "
                f"({summary['n_valid']} valid, {summary['n_hits']} hits), {summary['wall_hours']:.2f} h")
    return summary
