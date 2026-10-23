'''Best-effort end-of-run diagnostics for an active-learning scan: a few plots and a
text summary into ``<scan_dir>/diag/``. Never raises (a plotting hiccup must not fail a
completed scan, as for the emcee diagnostics). Torch-free; matplotlib only.'''
from __future__ import annotations

import json
import os

import numpy as np


def write_diagnostics(scan_dir: str, summary: dict, logger=None) -> None:
    diag = os.path.join(scan_dir, "diag")
    os.makedirs(diag, exist_ok=True)
    try:
        _write(diag, summary)
    except Exception as exc:   # best effort, by contract
        if logger:
            logger.warning(f"diagnostics failed (non-fatal): {exc!r}")


def _write(diag: str, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = summary.get("iterations", [])
    if not its:
        return
    it = np.array([i["iteration"] for i in its])
    n_lab = np.array([i["n_labelled"] for i in its])
    n_valid = np.array([i["n_valid"] for i in its])
    n_hits = np.array([i["n_hits"] for i in its])
    cum_lab = np.cumsum(n_lab)
    cum_hits = np.cumsum(n_hits)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit_rate = np.where(n_lab > 0, n_hits / n_lab, np.nan)
        valid_rate = np.where(n_lab > 0, n_valid / n_lab, np.nan)
        cum_rate = np.where(cum_lab > 0, cum_hits / cum_lab, np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(it, hit_rate, "o-", label="hits / labelled (per iteration)")
    ax[0].plot(it, valid_rate, "s--", label="valid / labelled (per iteration)")
    ax[0].plot(it, cum_rate, "-", label="hits / labelled (cumulative)")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("fraction")
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    ax[1].plot(cum_lab, cum_hits, "o-")
    ax[1].set_xlabel("models labelled (cumulative)")
    ax[1].set_ylabel("hits (cumulative)")
    fig.tight_layout()
    fig.savefig(os.path.join(diag, "hit_rate.png"), dpi=120)
    plt.close(fig)

    tr = [i.get("training") or {} for i in its]
    best_val = np.array([t.get("best_val_loss", np.nan) for t in tr], dtype=float)
    acc = np.array([t.get("verdict_accuracy_val", np.nan) for t in tr], dtype=float)
    n_train = np.array([t.get("n_train", 0) for t in tr], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(it, best_val, "o-")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(f"best validation loss ({tr[-1].get('loss_name', 'loss')})")
    ax[1].plot(it, acc, "o-", label="verdict accuracy (val)")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy")
    ax[1].set_ylim(0, 1)
    ax2 = ax[1].twinx()
    ax2.plot(it, n_train, "k:", label="training set size")
    ax2.set_ylabel("training set size")
    fig.tight_layout()
    fig.savefig(os.path.join(diag, "training.png"), dpi=120)
    plt.close(fig)

    lines = [
        f"active learning: {summary.get('model')} / {summary.get('head')} / {summary.get('strategy')}",
        f"iterations completed: {summary.get('iterations_done')} of {summary.get('n_iterations')}",
        f"models labelled: {int(cum_lab[-1])}, valid: {int(n_valid.sum())}, hits: {int(cum_hits[-1])}",
        f"cumulative hit rate: {cum_rate[-1]:.4f}",
        "",
        f"{'iter':>5} {'labelled':>9} {'valid':>6} {'hits':>5} {'hit rate':>9} {'|train|':>8} {'best val':>10} {'acc(val)':>9}",
    ]
    for k in range(len(its)):
        lines.append(f"{it[k]:>5} {n_lab[k]:>9} {n_valid[k]:>6} {n_hits[k]:>5} {hit_rate[k]:>9.4f} "
                     f"{int(n_train[k]):>8} {best_val[k]:>10.4g} {acc[k]:>9.4f}")
    with open(os.path.join(diag, "diagnostics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(diag, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
