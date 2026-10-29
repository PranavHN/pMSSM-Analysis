'''Candidate generation in the free parameter box and the labelled-set bookkeeping.

Torch-free. The parameter conventions are Run3ModelGen's: the 19 pMSSM parameters of the
config's ``parameters`` block, in the block's own order, with a parameter *fixed* when its
range is ``[v, v]`` (``ModelGenerator._is_fixed``) and *free* otherwise. The surrogate sees
the free parameters only; the fixed ones are filled in when a point is handed to the
pipeline.

Candidate pools follow the benchmark's ``pmssm.selection.generate_candidate_pool``: Latin
hypercube (``scipy.stats.qmc.LatinHypercube``) or uniform draws, scaled to the box, seeded
with ``seed * 10_000 + iteration`` (first attempt) and
``seed * 10_000 + iteration * 1000 + attempt`` (retries). The one difference is that the
hypercube is built over the free dimensions only; the benchmark drew all 19 and overwrote
the fixed columns, which is the same marginal distribution.
'''
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_MODELS_PER_ITERATION = 10000   # mirrors pipeline.MAX_MODELS_PER_JOB (one iteration = one scan dir)


def is_fixed(scanrange) -> bool:
    '''``[v, v]`` is fixed; interrupted ranges (list of sub-ranges) never are.'''
    if isinstance(scanrange[0], (list, tuple)):
        return False
    return scanrange[0] == scanrange[1]


def free_parameters(parameters: dict) -> list:
    '''Names of the free parameters, in the config's order.'''
    return [k for k, v in parameters.items() if not is_fixed(v)]


def fixed_parameters(parameters: dict) -> dict:
    return {k: float(v[0]) for k, v in parameters.items() if is_fixed(v)}


def _subranges(scanrange) -> list:
    '''A range as a list of ``(lo, hi)`` sub-ranges (one for a plain range).'''
    if isinstance(scanrange[0], (list, tuple)):
        return [(float(lo), float(hi)) for lo, hi in scanrange]
    return [(float(scanrange[0]), float(scanrange[1]))]


def free_box(parameters: dict, free_names=None) -> tuple:
    '''``(lo, hi)`` arrays of the free parameters' overall extent (min-max normalisation).'''
    names = free_names or free_parameters(parameters)
    lo = np.array([min(a for a, _ in _subranges(parameters[n])) for n in names])
    hi = np.array([max(b for _, b in _subranges(parameters[n])) for n in names])
    return lo, hi


def _map_unit(u: np.ndarray, scanrange) -> np.ndarray:
    '''Map uniform ``u`` in [0, 1) onto a (possibly interrupted) range, uniformly in
    length: sub-ranges are stacked by width, so the gaps carry no mass and a Latin
    hypercube stratification survives the mapping.'''
    subs = _subranges(scanrange)
    widths = np.array([hi - lo for lo, hi in subs])
    total = widths.sum()
    if total <= 0:
        return np.full_like(u, subs[0][0], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(widths) / total])
    out = np.empty_like(u, dtype=float)
    for (lo, hi), a, b in zip(subs, edges[:-1], edges[1:]):
        m = (u >= a) & (u < b) if b < 1.0 else (u >= a) & (u <= b)
        out[m] = lo + (u[m] - a) / (b - a) * (hi - lo)
    return out


def generate_candidates(parameters: dict, n: int, method: str = "lhs", seed: int = 0,
                        free_names=None) -> np.ndarray:
    '''``(n, n_free)`` candidate points in physical units.'''
    from scipy.stats import qmc

    names = free_names or free_parameters(parameters)
    d = len(names)
    if method == "lhs":
        unit = qmc.LatinHypercube(d=d, seed=int(seed)).random(n=int(n))
    elif method == "uniform":
        unit = np.random.default_rng(int(seed)).random((int(n), d))
    else:
        raise ValueError(f"unknown candidate generation {method!r}; use 'lhs' or 'uniform'")
    out = np.empty((int(n), d), dtype=float)
    for j, name in enumerate(names):
        out[:, j] = _map_unit(unit[:, j], parameters[name])
    return out


def to_points(free_values: np.ndarray, parameters: dict, free_names=None) -> dict:
    '''Expand ``(n, n_free)`` free values into the full ``{param: array}`` the pipeline
    needs (fixed parameters filled with their constant).'''
    names = free_names or free_parameters(parameters)
    free_values = np.asarray(free_values, dtype=float)
    n = free_values.shape[0]
    points = {}
    for name, scanrange in parameters.items():
        if name in names:
            points[name] = free_values[:, names.index(name)].copy()
        else:
            points[name] = np.full(n, float(scanrange[0]))
    return points


# ── the labelled set ────────────────────────────────────────────────────────────

ROLE_TRAIN = 0
ROLE_VAL = 1


@dataclass
class LabelledSet:
    '''Every model the loop has labelled, with the bookkeeping the ntuple merge and the
    diagnostics need. Rows are appended in labelling order; ``valid`` selects the rows
    the surrogate trains on.'''
    free_names: list
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))     # (N, n_free) physical units
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))          # raw target branch (-1 = not computed)
    valid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    hit: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    role: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    iteration: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    attempt: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    index: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # model index inside the iteration dir
    pred: np.ndarray = field(default_factory=lambda: np.zeros(0))       # surrogate prediction at selection (NaN for seeds)
    score: np.ndarray = field(default_factory=lambda: np.zeros(0))      # acquisition score at selection (NaN for seeds)

    def __post_init__(self):
        if self.X.shape[1:] == (0,):
            self.X = np.zeros((0, len(self.free_names)))

    def __len__(self):
        return int(self.X.shape[0])

    def append(self, X, y, valid, hit, role, iteration, attempt, index, pred=None, score=None):
        X = np.asarray(X, dtype=float).reshape(-1, len(self.free_names))
        n = X.shape[0]
        pred = np.full(n, np.nan) if pred is None else np.asarray(pred, dtype=float)
        score = np.full(n, np.nan) if score is None else np.asarray(score, dtype=float)
        self.X = np.concatenate([self.X, X])
        self.y = np.concatenate([self.y, np.asarray(y, dtype=float)])
        self.valid = np.concatenate([self.valid, np.asarray(valid, bool)])
        self.hit = np.concatenate([self.hit, np.asarray(hit, bool)])
        self.role = np.concatenate([self.role, np.asarray(role, np.int8)])
        self.iteration = np.concatenate([self.iteration, np.full(n, iteration, np.int32)])
        self.attempt = np.concatenate([self.attempt, np.full(n, attempt, np.int32)])
        self.index = np.concatenate([self.index, np.asarray(index, np.int32)])
        self.pred = np.concatenate([self.pred, pred])
        self.score = np.concatenate([self.score, score])

    def train_rows(self) -> np.ndarray:
        return np.where(self.valid & (self.role == ROLE_TRAIN))[0]

    def val_rows(self) -> np.ndarray:
        return np.where(self.valid & (self.role == ROLE_VAL))[0]

    def is_duplicate(self, X_new: np.ndarray) -> np.ndarray:
        '''Rows of ``X_new`` that already occur in the set (exact match), or that repeat
        an earlier row of ``X_new`` — the benchmark's two ``np.unique`` passes in one.'''
        X_new = np.asarray(X_new, dtype=float).reshape(-1, len(self.free_names))
        combined = np.concatenate([self.X, X_new]) if len(self) else X_new
        _, first = np.unique(combined, axis=0, return_index=True)
        keep = np.zeros(len(combined), bool)
        keep[first] = True
        return ~keep[len(self):]

    def to_dict(self) -> dict:
        return {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelledSet":
        return cls(**{k: (np.asarray(v) if k != "free_names" else list(v)) for k, v in d.items()})


def split_roles(n: int, val_fraction: float, rng: np.random.Generator) -> np.ndarray:
    '''Shuffle ``n`` new rows and give ``max(1, int(n * val_fraction))`` of them the
    validation role (the benchmark's per-batch 80/20 split). Returns the role array in
    the ORIGINAL row order; the caller shuffles rows with the same permutation if it
    wants the benchmark's exact ordering, which does not matter for training.'''
    role = np.full(n, ROLE_TRAIN, np.int8)
    if n == 0:
        return role
    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    perm = rng.permutation(n)
    role[perm[:n_val]] = ROLE_VAL
    return role
