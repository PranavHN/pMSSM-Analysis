'''The ``active_learning:`` config block — defaults, validation, target semantics.

Torch-free, so :class:`Run3ModelGen.modelgen.ModelGenerator` can validate a config in
the default environment (the way it validates ``constraints`` for ``prior: mcmc``) and
the test suite can cover every rule without a GPU.

Merging: :func:`Run3ModelGen.pipeline.load_scan_config` merges TOP-LEVEL keys only, so a
user config that sets ``active_learning: {model: dnn}`` would otherwise lose every other
default of the block. :func:`load_al_config` therefore deep-merges the user's block over
:data:`DEFAULTS`, which is the same dict ``default_config.yaml`` carries (a unit test pins
the two to each other).

Every value that used to be a click option of the benchmark drivers lives here; the key
names are the option names without the dashes. Defaults are the drivers' click defaults,
except where the paper's production ("campaign") values are used, which the comments in
``default_config.yaml`` mark.
'''
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

PRIOR_NAME = "active_learning"

MODELS = ("transformer", "dnn", "exact_gp", "deep_gp")
NEURAL_MODELS = ("transformer", "dnn")
GP_MODELS = ("exact_gp", "deep_gp")
HEADS = ("classification", "regression")
# Which acquisition rules a head can drive. classification: entropy only; regression:
# the entropy batch (production) and the tolerance-cut + random control arm.
STRATEGIES_BY_HEAD = {
    "classification": ("cls_entropy",),
    "regression": ("entropy_batch", "tol_only_random"),
}
STRATEGIES = tuple(s for ss in STRATEGIES_BY_HEAD.values() for s in ss)
ENVIRONMENTS = ("rocm", "cuda")          # never "default": the loop has no CPU mode
CANDIDATE_GENERATION = ("lhs", "uniform")
TRANSFORMS = ("log", "none")
GP_KERNELS = ("RBF", "Matern", "RQK")

# The block as default_config.yaml ships it. Keep the two in sync (pinned by
# tests/test_active_learning_config.py).
DEFAULTS: dict = {
    "environment": "rocm",
    "nworkers": 16,
    "walltime_hours": 23,
    "seed_models": 2000,
    "n_iterations": 40,
    "val_fraction": 0.2,
    "model": "transformer",
    "head": "classification",
    "strategy": "cls_entropy",
    "target": {
        "branch": "MO_Omega",
        "value": 0.12,
        "transform": "log",
        "valid": {"lo": 0.0, "hi": 1.0},
        "hit_tolerance": 0.10,
    },
    "vetoes": [
        {"branch": "SP_m_h", "lo": 0.0},
        {"branch": "SP_LSP_type", "lo": 0.5, "hi": 3.5},
    ],
    "candidates": {"n_candidates": 1000000, "generation": "lhs"},
    "training": {
        "epochs": 10000,
        "patience": 200,
        "warm_start": False,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "batch_size": 256,
    },
    "transformer": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 512,
                    "dropout": 0.1},
    "dnn": {"d_model": 64, "num_layers": 4, "dim_feedforward": 256, "dropout": 0.1},
    "gp": {
        "kernel": "RBF",
        "lengthscale": 1.0,
        "noise": 0.01,
        "jitter": 0.001,
        "use_ard": True,
        "learning_rate": 0.001,
        "batch_size": 256,
        "num_samples": 8,
        "num_hidden_dims": 10,
        "num_inducing_max": 256,
    },
    "acquisition": {
        "mc_samples": 30,
        "tolerance": 1.0,
        "proximity": 0.1,
        "entropy_pool_size": 5000,
        "entropy_blur": 0.15,
        "entropy_beta": 50.0,
    },
    "labelling": {"min_valid_fraction": 0.6, "max_attempts": 10},
}


class AlConfigError(ValueError):
    '''A rejected ``active_learning:`` block. The message names the key.'''


def deep_merge(base: Mapping, override: Mapping | None) -> dict:
    '''Recursive dict merge: ``override`` wins, nested dicts merge, lists replace.'''
    out = copy.deepcopy(dict(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# ── target and vetoes ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Veto:
    '''A strict window on one ntuple branch; a model is valid only if every veto passes.

    Same semantics as the ``hard_window`` constraint type: ``lo < value < hi``, each bound
    optional. The ntuple's ``-1`` "not filled" sentinel fails any ``lo >= -1``, which is
    what makes ``{branch: SP_m_h, lo: 0}`` read as "SPheno produced a spectrum".'''
    branch: str
    lo: float | None = None
    hi: float | None = None

    def passes(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        ok = np.isfinite(v)
        if self.lo is not None:
            ok &= v > self.lo
        if self.hi is not None:
            ok &= v < self.hi
        return ok


@dataclass(frozen=True)
class TargetSpec:
    '''What the surrogate learns: one ntuple branch, its target value, and the band.

    ``transform: log`` puts the model in ``t = log(Y / value)`` space, so the target value
    sits at ``t = 0`` (the benchmark's convention): the classification label is ``Y > value``
    (``t > 0``), regression acquisition is anchored at ``t = 0``. ``transform: none`` leaves
    ``t = Y`` with the threshold at ``value``.'''
    branch: str
    value: float
    transform: str = "log"
    valid_lo: float | None = None
    valid_hi: float | None = None
    hit_tolerance: float = 0.10

    @property
    def threshold(self) -> float:
        '''The decision point in transformed space.'''
        return 0.0 if self.transform == "log" else float(self.value)

    def transform_y(self, y):
        y = np.asarray(y, dtype=float)
        if self.transform == "log":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log(y / self.value)
        return y.copy()

    def inverse_transform_y(self, t):
        t = np.asarray(t, dtype=float)
        if self.transform == "log":
            return self.value * np.exp(t)
        return t.copy()

    def is_valid(self, y: np.ndarray) -> np.ndarray:
        '''Finite, inside the ``valid`` window, and positive when the transform is log.'''
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if self.transform == "log":
            ok &= y > 0
        if self.valid_lo is not None:
            ok &= y > self.valid_lo
        if self.valid_hi is not None:
            ok &= y < self.valid_hi
        return ok

    def is_hit(self, y: np.ndarray) -> np.ndarray:
        '''Inside the relative band ``|Y - value| / |value| < hit_tolerance`` (and valid).'''
        y = np.asarray(y, dtype=float)
        return self.is_valid(y) & (np.abs(y - self.value) / abs(self.value) < self.hit_tolerance)


@dataclass(frozen=True)
class AlConfig:
    '''The validated ``active_learning:`` block plus the top-level keys the loop reads.'''
    environment: str
    nworkers: int
    walltime_hours: float
    seed_models: int
    n_iterations: int
    num_models: int                      # top-level num_models: selected + labelled per iteration
    val_fraction: float
    model: str
    head: str
    strategy: str
    target: TargetSpec
    vetoes: tuple[Veto, ...]
    n_candidates: int
    candidate_generation: str
    training: dict
    transformer: dict
    dnn: dict
    gp: dict
    acquisition: dict
    labelling: dict
    raw: dict = field(default_factory=dict, repr=False)   # the merged block, for provenance

    @property
    def is_gp(self) -> bool:
        return self.model in GP_MODELS

    @property
    def is_neural(self) -> bool:
        return self.model in NEURAL_MODELS

    @property
    def threshold(self) -> float:
        return self.target.threshold


# ── validation ──────────────────────────────────────────────────────────────────

def _fail(key: str, msg: str):
    raise AlConfigError(f"active_learning.{key}: {msg}")


def _number(block: Mapping, key: str, path: str, *, kind=float, lo=None, hi=None,
            lo_strict=False, hi_strict=False):
    if key not in block or block[key] is None:
        _fail(path, "missing")
    try:
        v = kind(block[key])
    except (TypeError, ValueError):
        _fail(path, f"expected a {kind.__name__}, got {block[key]!r}")
    if kind is int and isinstance(block[key], float) and not float(block[key]).is_integer():
        _fail(path, f"expected an integer, got {block[key]!r}")
    if not math.isfinite(v):
        _fail(path, f"must be finite, got {v!r}")
    if lo is not None and (v <= lo if lo_strict else v < lo):
        _fail(path, f"must be {'>' if lo_strict else '>='} {lo}, got {v}")
    if hi is not None and (v >= hi if hi_strict else v > hi):
        _fail(path, f"must be {'<' if hi_strict else '<='} {hi}, got {v}")
    return v


def _choice(block: Mapping, key: str, path: str, choices):
    v = block.get(key)
    if v not in choices:
        _fail(path, f"must be one of {list(choices)}, got {v!r}")
    return v


def _optional_number(block: Mapping, key: str, path: str):
    v = block.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        _fail(path, f"expected a number or null, got {v!r}")
    if not math.isfinite(f):
        _fail(path, f"must be finite, got {f!r}")
    return f


def _branch_name(v, path: str) -> str:
    if not isinstance(v, str) or not v.strip():
        _fail(path, f"expected a non-empty ntuple branch name, got {v!r}")
    return v.strip()


def _parse_target(block: Mapping) -> TargetSpec:
    if not isinstance(block, Mapping):
        _fail("target", "must be a mapping with branch/value/transform/valid/hit_tolerance")
    branch = _branch_name(block.get("branch"), "target.branch")
    transform = _choice(block, "transform", "target.transform", TRANSFORMS)
    value = _number(block, "value", "target.value")
    if transform == "log" and value <= 0:
        _fail("target.value", f"must be > 0 with transform 'log' (t = log(Y / value)), got {value}")
    if value == 0:
        _fail("target.value", "must be non-zero (the hit band is relative to it)")
    valid = block.get("valid") or {}
    if not isinstance(valid, Mapping):
        _fail("target.valid", f"must be a mapping {{lo, hi}} or null, got {valid!r}")
    lo = _optional_number(valid, "lo", "target.valid.lo")
    hi = _optional_number(valid, "hi", "target.valid.hi")
    if lo is not None and hi is not None and lo >= hi:
        _fail("target.valid", f"empty window lo={lo} >= hi={hi}")
    tol = _number(block, "hit_tolerance", "target.hit_tolerance", lo=0, lo_strict=True)
    return TargetSpec(branch=branch, value=value, transform=transform,
                      valid_lo=lo, valid_hi=hi, hit_tolerance=tol)


def _parse_vetoes(items) -> tuple[Veto, ...]:
    if items is None:
        return ()
    if not isinstance(items, (list, tuple)):
        _fail("vetoes", f"must be a list of {{branch, lo, hi}} entries, got {items!r}")
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, Mapping):
            _fail(f"vetoes[{i}]", f"must be a mapping, got {item!r}")
        branch = _branch_name(item.get("branch"), f"vetoes[{i}].branch")
        lo = _optional_number(item, "lo", f"vetoes[{i}].lo")
        hi = _optional_number(item, "hi", f"vetoes[{i}].hi")
        if lo is None and hi is None:
            _fail(f"vetoes[{i}]", f"'{branch}' sets neither lo nor hi; it would veto nothing")
        if lo is not None and hi is not None and lo >= hi:
            _fail(f"vetoes[{i}]", f"'{branch}' has an empty window lo={lo} >= hi={hi}")
        out.append(Veto(branch=branch, lo=lo, hi=hi))
    return tuple(out)


def _sub(block: Mapping, key: str) -> Mapping:
    v = block.get(key)
    if v is None:
        return {}
    if not isinstance(v, Mapping):
        _fail(key, f"must be a mapping, got {v!r}")
    return v


def load_al_config(merged: Mapping[str, Any]) -> AlConfig:
    '''Validate a MERGED scan config (output of ``pipeline.load_scan_config``).

    Raises :class:`AlConfigError` naming the offending key. The user's
    ``active_learning`` block is deep-merged over :data:`DEFAULTS` first, so partial
    blocks are fine.'''
    from Run3ModelGen import pipeline   # local: keeps this module importable standalone

    user_block = merged.get("active_learning")
    if user_block is not None and not isinstance(user_block, Mapping):
        _fail("", f"the active_learning block must be a mapping, got {user_block!r}")
    al = deep_merge(DEFAULTS, user_block)

    environment = _choice(al, "environment", "environment", ENVIRONMENTS + ("default",))
    if environment == "default":
        _fail("environment", "the default environment is CPU-only and prior: active_learning "
              "has no CPU mode; set it to 'rocm' or 'cuda' and run from that environment "
              "(pixi install -e rocm && pixi run -e rocm ...)")

    nworkers = _number(al, "nworkers", "nworkers", kind=int, lo=1)
    walltime = _number(al, "walltime_hours", "walltime_hours", lo=0, lo_strict=True)
    seed_models = _number(al, "seed_models", "seed_models", kind=int, lo=1,
                          hi=pipeline.MAX_MODELS_PER_JOB)
    n_iterations = _number(al, "n_iterations", "n_iterations", kind=int, lo=0)
    val_fraction = _number(al, "val_fraction", "val_fraction", lo=0, hi=1,
                           lo_strict=True, hi_strict=True)

    num_models = merged.get("num_models")
    if num_models is None:
        _fail("", "top-level num_models (points selected and labelled per iteration) is missing")
    num_models = _number({"num_models": num_models}, "num_models", "num_models (top-level)",
                         kind=int, lo=1, hi=pipeline.MAX_MODELS_PER_JOB)

    model = _choice(al, "model", "model", MODELS)
    head = _choice(al, "head", "head", HEADS)
    strategy = al.get("strategy")
    allowed = STRATEGIES_BY_HEAD[head]
    if strategy not in allowed:
        hint = (" (cls_entropy is the only classification rule)" if head == "classification"
                else " (regression rules: entropy_batch, tol_only_random)")
        _fail("strategy", f"{strategy!r} does not go with head {head!r}{hint}")

    target = _parse_target(_sub(al, "target"))
    vetoes = _parse_vetoes(al.get("vetoes"))

    cand = _sub(al, "candidates")
    n_candidates = _number(cand, "n_candidates", "candidates.n_candidates", kind=int, lo=1)
    generation = _choice(cand, "generation", "candidates.generation", CANDIDATE_GENERATION)

    tr = dict(_sub(al, "training"))
    tr["epochs"] = _number(tr, "epochs", "training.epochs", kind=int, lo=1)
    tr["patience"] = _number(tr, "patience", "training.patience", kind=int, lo=1)
    tr["warm_start"] = bool(tr.get("warm_start", False))
    tr["learning_rate"] = _number(tr, "learning_rate", "training.learning_rate", lo=0, lo_strict=True)
    tr["weight_decay"] = _number(tr, "weight_decay", "training.weight_decay", lo=0)
    tr["batch_size"] = _number(tr, "batch_size", "training.batch_size", kind=int, lo=1)

    tf = dict(_sub(al, "transformer"))
    for k in ("d_model", "nhead", "num_layers", "dim_feedforward"):
        tf[k] = _number(tf, k, f"transformer.{k}", kind=int, lo=1)
    tf["dropout"] = _number(tf, "dropout", "transformer.dropout", lo=0, hi=1, hi_strict=True)
    if tf["d_model"] % tf["nhead"]:
        _fail("transformer", f"d_model={tf['d_model']} must be divisible by nhead={tf['nhead']}")

    dnn = dict(_sub(al, "dnn"))
    for k in ("d_model", "num_layers", "dim_feedforward"):
        dnn[k] = _number(dnn, k, f"dnn.{k}", kind=int, lo=1)
    dnn["dropout"] = _number(dnn, "dropout", "dnn.dropout", lo=0, hi=1, hi_strict=True)
    if model in NEURAL_MODELS and {"transformer": tf, "dnn": dnn}[model]["dropout"] == 0:
        _fail(f"{model}.dropout", "must be > 0: MC dropout is the uncertainty of the "
              "neural surrogates, and every strategy here needs it")

    gp = dict(_sub(al, "gp"))
    gp["kernel"] = _choice(gp, "kernel", "gp.kernel", GP_KERNELS)
    gp["lengthscale"] = _number(gp, "lengthscale", "gp.lengthscale", lo=0, lo_strict=True)
    gp["noise"] = _number(gp, "noise", "gp.noise", lo=0, lo_strict=True)
    gp["jitter"] = _number(gp, "jitter", "gp.jitter", lo=0, lo_strict=True)
    gp["use_ard"] = bool(gp.get("use_ard", True))
    gp["learning_rate"] = _number(gp, "learning_rate", "gp.learning_rate", lo=0, lo_strict=True)
    gp["batch_size"] = _number(gp, "batch_size", "gp.batch_size", kind=int, lo=1)
    gp["num_samples"] = _number(gp, "num_samples", "gp.num_samples", kind=int, lo=1)
    gp["num_hidden_dims"] = _number(gp, "num_hidden_dims", "gp.num_hidden_dims", kind=int, lo=1)
    gp["num_inducing_max"] = _number(gp, "num_inducing_max", "gp.num_inducing_max", kind=int, lo=2)

    acq = dict(_sub(al, "acquisition"))
    acq["mc_samples"] = _number(acq, "mc_samples", "acquisition.mc_samples", kind=int, lo=2)
    acq["tolerance"] = _number(acq, "tolerance", "acquisition.tolerance", lo=0)
    acq["proximity"] = _number(acq, "proximity", "acquisition.proximity", lo=0)
    acq["entropy_pool_size"] = _number(acq, "entropy_pool_size", "acquisition.entropy_pool_size",
                                       kind=int, lo=1)
    acq["entropy_blur"] = _number(acq, "entropy_blur", "acquisition.entropy_blur", lo=0)
    acq["entropy_beta"] = _number(acq, "entropy_beta", "acquisition.entropy_beta", lo=0)

    lab = dict(_sub(al, "labelling"))
    lab["min_valid_fraction"] = _number(lab, "min_valid_fraction", "labelling.min_valid_fraction",
                                        lo=0, hi=1, lo_strict=True)
    lab["max_attempts"] = _number(lab, "max_attempts", "labelling.max_attempts", kind=int, lo=1)

    return AlConfig(
        environment=environment, nworkers=nworkers, walltime_hours=walltime,
        seed_models=seed_models, n_iterations=n_iterations, num_models=num_models,
        val_fraction=val_fraction, model=model, head=head, strategy=strategy,
        target=target, vetoes=vetoes, n_candidates=n_candidates,
        candidate_generation=generation, training=tr, transformer=tf, dnn=dnn, gp=gp,
        acquisition=acq, labelling=lab, raw=al,
    )


def validate_active_learning(merged: Mapping[str, Any]) -> None:
    '''Raise :class:`AlConfigError` if ``merged`` (with ``prior: active_learning``) is invalid.

    Called by ``ModelGenerator.__init__`` so a bad block fails at config load, before
    anything is submitted — the same contract ``_validate_observable_sources`` gives
    ``prior: mcmc``.'''
    load_al_config(merged)
