'''Shared HDF5-backend -> ROOT "susy" ntuple conversion for the emcee scan.

Single source of truth used by BOTH:
  * the sampler (`Run3ModelGen.emcee_sampler`), which writes the ntuple at the
    end of a run, and
  * the diagnostics script (`scripts/emcee_diagnostics.py`, `--to-ntuple`),
    which regenerates it from `chain.h5` with a chosen discard/thin.

Kept free of any physics / ModelGenerator imports (numpy + json + uproot + emcee
only) so the diagnostics can pull it without dragging the full package.

The IN_* / observable branch layout comes from the backend's sidecar meta
(`in_branches`, `record_branches`), which the sampler computes once (see
`emcee_sampler._build_in_branch_layout`) and stores in `<backend>.meta.json`.
'''
from __future__ import annotations

import json
import os

import numpy as np
import emcee


def resolve_burnin(max_tau, n_iter, burn_in_floor=0, factor=3.0):
    '''Number of initial iterations to discard as burn-in — the SINGLE SOURCE OF
    TRUTH shared by the sampler (`emcee_sampler`, which writes the ntuple at the
    end of a run) and the diagnostics (`scripts/emcee_diagnostics.py`), so both
    always discard the same iterations.

    discard = max(burn_in_floor, ceil(factor * max_tau)), capped at half the
    chain. When max_tau is not finite (autocorrelation estimate unavailable) only
    the absolute floor applies. `burn_in_floor` and `factor` are carried in the
    backend's .meta.json sidecar (keys 'burn_in' and 'burn_in_factor') so both
    sides read identical values.'''
    tau_burn = int(np.ceil(factor * max_tau)) if np.isfinite(max_tau) else 0
    discard = max(int(burn_in_floor), tau_burn)
    return max(0, min(discard, n_iter // 2))


def build_ntuple_tree(backend_path, meta: dict, discard=0, thin=1):
    '''Build the "susy" tree dict + sample count from an HDF5 backend using the
    sidecar `meta` layout (in_branches / record_branches).'''
    backend = emcee.backends.HDFBackend(backend_path, read_only=True)
    chain = backend.get_chain(discard=discard, thin=thin, flat=True)    # (S, ndim)
    logp = backend.get_log_prob(discard=discard, thin=thin, flat=True)  # (S,)
    try:
        blobs = backend.get_blobs(discard=discard, thin=thin, flat=True)
    except Exception:
        blobs = None
    n = chain.shape[0]

    # emcee_logp / chi2 are written unconditionally below, so no other branch may
    # claim those names, and no two branches (IN_* or record_observables) may share
    # one — a dict assignment would silently overwrite, corrupting the ntuple. Route
    # every non-reserved branch through _assign so a collision fails loudly instead.
    RESERVED = ('emcee_logp', 'chi2')
    tree = {}

    def _assign(branch, values, origin):
        if branch in RESERVED or branch in tree:
            raise ValueError(
                f"ntuple branch name collision on {branch!r} (from {origin}): it is "
                f"reserved ({', '.join(RESERVED)}) or already used by another branch. "
                f"Rename the colliding record_observables 'branch' in the config.")
        tree[branch] = values

    for b in meta['in_branches']:
        if b['kind'] == 'free':
            _assign(b['branch'], chain[:, b['index']].astype(np.float64), 'in_branches')
        else:
            _assign(b['branch'], np.full(n, b['value'], dtype=np.float64), 'in_branches')

    if blobs is not None:
        for rb in meta.get('record_branches', []):
            arr = np.asarray(blobs[rb['name']], dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, -1.0)  # match mkntuple -1 sentinel
            _assign(rb['branch'], arr, 'record_observables')

    tree['emcee_logp'] = logp.astype(np.float64)
    tree['chi2'] = (-2.0 * logp).astype(np.float64)
    return tree, n


def h5_to_ntuple(backend_path, out_root, discard=0, thin=1, meta=None):
    '''Convert an emcee HDF5 backend to a ROOT "susy" ntuple with the exact
    IN_* / observable branch names used by the rest of Run3ModelGen, so that
    extractModels and other ntuple consumers keep working. Reads the sidecar
    `<backend_path>.meta.json` unless `meta` is supplied.

    Returns (n_samples, n_branches). Logging is left to the caller so each side
    can use its own logger.'''
    import uproot
    if meta is None:
        with open(backend_path + '.meta.json') as f:
            meta = json.load(f)
    tree, n = build_ntuple_tree(backend_path, meta, discard=discard, thin=thin)
    os.makedirs(os.path.dirname(os.path.abspath(out_root)) or '.', exist_ok=True)
    with uproot.recreate(out_root) as f:
        f["susy"] = tree
    return n, len(tree)
