'''Shared helpers for the CI scan reference files.

Single source of truth for:
  * the CI scan settings (:data:`SCANS`),
  * where their reference files live on EOS (:func:`ntuple_url`/:func:`counts_url`),
  * how to compute per-step success counts for a finished scan
    (:func:`per_step_counts`), and
  * how to compare produced counts against a reference (:func:`compare_counts`).

Used by both ``make_references.py`` (produce + upload references) and
``compare_reference.py`` (per-step CI regression check). Importing this module has
no side effects (``uproot`` is imported lazily inside :func:`per_step_counts`).
'''

import os
import glob

# Shared EOS location (xrootd URL) for the reference files.
EOS_BASE = "root://eosuser.cern.ch//eos/user/n/nicebot/Run3ModelGen"

# Path (relative to the run/ working dir, matching the CI invocations) of the
# importance seed CSV that the importance config references by bare name.
SEED_CSV = "../source/Run3ModelGen/data/configs/ci_importance_seeds.csv"

# The scan settings exercised in CI. ``config`` is given relative to the run/ dir
# so it matches the CI ``genModels.py`` invocations byte-for-byte; ``None`` means
# the default config. ``needs_seed_csv`` marks scans whose config references
# ci_importance_seeds.csv by bare filename (must be copied into run/ first).
SCANS = [
    {"name": "flat",       "config": None,                                                            "scan_dir": "scan_flat",       "needs_seed_csv": False},
    {"name": "fixed",      "config": "../source/Run3ModelGen/data/configs/example_fixed_config.yaml", "scan_dir": "scan_fixed",      "needs_seed_csv": False},
    {"name": "mcmc",       "config": "../source/Run3ModelGen/data/configs/ci_mcmc_config.yaml",       "scan_dir": "scan_mcmc",       "needs_seed_csv": False},
    {"name": "grid",       "config": "../source/Run3ModelGen/data/configs/ci_grid_config.yaml",       "scan_dir": "scan_grid",       "needs_seed_csv": False},
    {"name": "importance", "config": "../source/Run3ModelGen/data/configs/ci_importance_config.yaml", "scan_dir": "scan_importance", "needs_seed_csv": True},
]

# Representative ntuple branch per pipeline step. A model "passed" the step if the
# branch value != -1 (the ntuple's sentinel for "not filled"). This is the same
# idiom used by extractModels (default selection ``SS_m_h != -1``). SModelS is
# handled separately (see below): its ntuple branches are only filled when there
# is a non-empty result, so counting them would undercount successes.
STEP_BRANCH = {
    "prep_input": "IN_M_1",
    "SPheno":     "SP_m_h",
    "softsusy":   "SS_m_h",
    "micromegas": "MO_Omega",
    "superiso":   "SI_BR_b_to_sgamma",
    "gm2calc":    "GM2_gmuon",
    "feynhiggs":  "FH_m_h",
}

# Canonical display/compare order (SModelS, counted from files, last).
STEP_ORDER = list(STEP_BRANCH.keys()) + ["SModelS"]


def by_name(name):
    '''Return the SCANS entry with the given name (raises KeyError if unknown).'''
    for scan in SCANS:
        if scan["name"] == name:
            return scan
    raise KeyError(f"Unknown scan '{name}'. Known: {[s['name'] for s in SCANS]}")


def ntuple_path(scan_dir):
    '''Local path of the ntuple produced inside a scan directory.'''
    return f"{scan_dir}/ntuple.0.0.root"


def ntuple_url(name):
    '''EOS xrootd URL of the reference ntuple for a scan.'''
    return f"{EOS_BASE}/ntuple_{name}.root"


def counts_url(name):
    '''EOS xrootd URL of the reference per-step counts JSON for a scan.'''
    return f"{EOS_BASE}/counts_{name}.json"


def per_step_counts(scan_dir):
    '''Compute per-step success counts for a finished scan directory.

    :param scan_dir: path to a ``scan_<name>`` directory (containing
        ``ntuple.0.0.root`` and the per-step output subdirs).
    :returns: ``{step: (passed, total)}``. Steps whose representative branch is
        absent from the ntuple (reduced scans) are omitted. SModelS is included
        only if a ``SModelS/`` subdir exists, counted from its output files.
    '''
    import uproot  # lazy: keeps importing this module cheap/side-effect free

    counts = {}
    with uproot.open(f"{ntuple_path(scan_dir)}:susy") as tree:
        total = int(tree.num_entries)
        available = set(tree.keys())
        present = {step: br for step, br in STEP_BRANCH.items() if br in available}
        arrs = tree.arrays(list(present.values()), library="np") if present else {}
        for step, branch in present.items():
            counts[step] = (int((arrs[branch] != -1).sum()), total)

    # SModelS: count per-model output files (written only on success), not an
    # ntuple branch (those are only filled when the result is non-empty).
    smodels_dir = os.path.join(scan_dir, "SModelS")
    if os.path.isdir(smodels_dir):
        passed = len(glob.glob(os.path.join(smodels_dir, "*.slha.py")))
        counts["SModelS"] = (passed, total)

    return counts


def format_counts_table(counts, title=None):
    '''Return a human-readable per-step table for a single counts dict.'''
    lines = [title] if title else []
    for step in STEP_ORDER:
        if step in counts:
            passed, total = counts[step]
            lines.append(f"    {step:<12} {passed}/{total}")
    return "\n".join(lines)


def compare_counts(ref, produced):
    '''Compare two ``{step: (passed, total)}`` dicts.

    :returns: ``(ok, table_str)``. ``ok`` is False if any step is present in only
        one dict or if any ``(passed, total)`` differs.
    '''
    steps = [s for s in STEP_ORDER if s in ref or s in produced]
    for s in list(ref) + list(produced):  # keep any unexpected extras, deterministically
        if s not in steps:
            steps.append(s)

    ok = True
    rows = [f"    {'step':<12} {'reference':>11} {'produced':>11}   status"]
    for step in steps:
        r = ref.get(step)
        p = produced.get(step)
        status = "OK"
        if r != p:
            ok = False
            status = "DIFF"
        r_str = f"{r[0]}/{r[1]}" if r is not None else "-"
        p_str = f"{p[0]}/{p[1]}" if p is not None else "-"
        rows.append(f"    {step:<12} {r_str:>11} {p_str:>11}   {status}")
    return ok, "\n".join(rows)


def counts_to_json(counts):
    '''Serialize ``{step: (passed, total)}`` to JSON-friendly ``{step: [passed, total]}``.'''
    return {step: [int(p), int(t)] for step, (p, t) in counts.items()}


def counts_from_json(obj):
    '''Inverse of :func:`counts_to_json`.'''
    return {step: (int(v[0]), int(v[1])) for step, v in obj.items()}
