'''Merge the per-iteration ntuples of an active-learning scan into the scan's ntuple.

Torch-free. The result is the ordinary ``ntuple.<C>.<P>.root`` every prior produces (one
``susy`` tree, flat double branches, ``-1`` for "not computed", ``model`` ids following the
``b * 100000 + index`` batch contract) plus the ``al_*`` branches below, in the spirit of
the emcee ntuple's ``emcee_logp``/``chi2``. Everything is rebuilt from what the iteration
dirs hold (``ntuple.0.0.root`` + ``selection.json``), so the merge can be redone by hand.
'''
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

AL_BRANCHES = {
    "al_iteration": "iteration that labelled the model (0 = the uniform seed set)",
    "al_attempt": "labelling attempt within the iteration (0 = the selected batch)",
    "al_index": "model index inside iteration_NNN/ (its id in that ntuple)",
    "al_valid": "1 if the target window and every veto passed, else 0",
    "al_hit": "1 if inside the hit band |Y - value| / |value| < hit_tolerance, else 0",
    "al_pred": "surrogate prediction in transformed space when the point was selected (-1 for seeds)",
    "al_score": "acquisition score when the point was selected (-1 for seeds / tol_only_random)",
}
SELECTION_JSON = "selection.json"
ITERATION_RE = re.compile(r"^iteration_(\d{3,})$")


def iteration_dirs(scan_dir: str) -> list:
    '''``iteration_NNN`` dirs of a scan, in iteration order.'''
    out = []
    for name in os.listdir(scan_dir):
        m = ITERATION_RE.match(name)
        if m and os.path.isdir(os.path.join(scan_dir, name)):
            out.append((int(m.group(1)), os.path.join(scan_dir, name)))
    return [d for _, d in sorted(out)]


def iteration_number(iteration_dir: str) -> int:
    return int(ITERATION_RE.match(os.path.basename(iteration_dir.rstrip("/"))).group(1))


def _selection_columns(iteration_dir: str, n: int) -> tuple:
    '''``(attempt, pred, score)`` per model index from ``selection.json`` (-1 defaults).'''
    attempt = np.zeros(n, dtype=float)
    pred = np.full(n, -1.0)
    score = np.full(n, -1.0)
    path = os.path.join(iteration_dir, SELECTION_JSON)
    if not os.path.exists(path):
        return attempt, pred, score
    with open(path) as f:
        doc = json.load(f)
    for row in doc.get("models", []):
        i = int(row["index"])
        if i >= n:
            continue
        attempt[i] = float(row.get("attempt", 0))
        p, s = row.get("pred"), row.get("score")
        pred[i] = -1.0 if p is None or not np.isfinite(p) else float(p)
        score[i] = -1.0 if s is None or not np.isfinite(s) else float(s)
    return attempt, pred, score


def merge_iterations(scan_dir: str, out_root: str, target, vetoes, batch_index: int = 0,
                     logger=None) -> tuple:
    '''Write the merged ntuple; returns ``(n_models, n_branches)``.

    ``model = batch_index * 100000 + running index`` over all iterations in order (the
    batch contract of ``ntupling.readModel``; ``batch_index`` is ``b`` of ``scan.<C>.<b>``,
    0 for a single scan).'''
    import uproot

    dirs = iteration_dirs(scan_dir)
    if not dirs:
        raise RuntimeError(f"no iteration_NNN directories under {scan_dir}")

    # pass 1: branch union and sizes
    sizes, branch_set = [], set()
    for d in dirs:
        path = os.path.join(d, "ntuple.0.0.root")
        if not os.path.exists(path):
            raise RuntimeError(f"{d} has no ntuple.0.0.root; the iteration is incomplete")
        with uproot.open(path) as f:
            tree = f["susy"]
            sizes.append(int(tree.num_entries))
            branch_set.update(tree.keys())
    branch_set.discard("model")
    total = int(sum(sizes))
    tree_out = {b: np.full(total, -1.0) for b in sorted(branch_set)}
    for b in AL_BRANCHES:
        tree_out[b] = np.full(total, -1.0)
    model = np.empty(total, dtype=float)

    # pass 2: fill
    start = 0
    for d, n in zip(dirs, sizes):
        it = iteration_number(d)
        with uproot.open(os.path.join(d, "ntuple.0.0.root")) as f:
            arrays = f["susy"].arrays(library="np")
        sl = slice(start, start + n)
        for b, values in arrays.items():
            if b == "model":
                continue
            tree_out[b][sl] = np.asarray(values, dtype=float)
        y = np.asarray(arrays.get(target.branch, np.full(n, -1.0)), dtype=float)
        valid = target.is_valid(y)
        for veto in vetoes:
            valid &= veto.passes(np.asarray(arrays.get(veto.branch, np.full(n, -1.0)), dtype=float))
        attempt, pred, score = _selection_columns(d, n)
        tree_out["al_iteration"][sl] = it
        tree_out["al_attempt"][sl] = attempt
        tree_out["al_index"][sl] = np.arange(n)
        tree_out["al_valid"][sl] = valid.astype(float)
        tree_out["al_hit"][sl] = (valid & target.is_hit(y)).astype(float)
        tree_out["al_pred"][sl] = pred
        tree_out["al_score"][sl] = score
        model[sl] = batch_index * 100000 + np.arange(start, start + n)
        start += n

    tree = {"model": model}
    tree.update(tree_out)
    os.makedirs(os.path.dirname(os.path.abspath(out_root)) or ".", exist_ok=True)
    with uproot.recreate(out_root) as f:
        f["susy"] = tree
    if logger:
        logger.info(f"merged {len(dirs)} iterations, {total} models, {len(tree)} branches -> {out_root}")
    return total, len(tree)
