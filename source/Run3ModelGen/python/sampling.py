'''Point-sampling functions for the flat/fixed/grid/importance priors.

Bodies are lifted VERBATIM from the ModelGenerator.sample_* methods
(modelgen.py) — the numpy draw order must not change: for a given
np.random.seed the sampled points are byte-identical to the pre-refactor
code, which is what keeps the CI per-step reference counts valid.
ModelGenerator.sample_* are now one-line delegates into this module.

The caller owns the RNG state: np.random.seed(seed) is applied by
ModelGenerator.__init__ (or by the Snakemake sample rule) BEFORE calling
into this module, and nothing here reseeds.

The MCMC prior (prior: mcmc) is the emcee ensemble sampler
(Run3ModelGen.emcee_sampler): it runs the whole pipeline inside sampling and is
executed as one opaque chain job, not as a point sampler here.
'''

import os
import numpy as np

import structlog
log = structlog.get_logger()


def sample_flat(parameters: dict, num_models: int) -> dict:
    '''Sample parameters uniformly at random within the config ranges.
    Supports interrupted (gapped) ranges. Returns {param: np.ndarray}.'''

    points = {}
    for key, scanrange in parameters.items():

        # Check if the provided range is interrupted
        if type(scanrange[0]) == list:
            weights = []
            dist = 0

            # Loop over ranges to get weights for numbers of models
            for subrange in scanrange:
                if type(subrange) != list: raise ValueError(f"Provided scanrange for parameter {key} is not supported!")
                weights.append((subrange[1]-subrange[0]))
                dist += subrange[1]-subrange[0]

            weights = [weight/dist for weight in weights]

            # Generate random points according to weights and append result in one np array
            # Making sure that overall number of models matches
            points_left = num_models
            keypoints = np.array([])
            list_numpoints = []
            for i in range(len(scanrange)):
                numpoints = int(num_models*weights[i])
                if i == len(scanrange)-1: numpoints = points_left
                list_numpoints += [numpoints]

                keypoints = np.append(keypoints, np.random.uniform(scanrange[i][0], scanrange[i][1], numpoints))
                points_left -= numpoints

            points[key] = keypoints

            log.info(f"Found gap in scanning ranges for parameter {key}. Generated random numbers for subranges with weights:", weights=weights)
            log.info("Resulting in these numbers of points:", list_numpoints=list_numpoints)

        else:
            points[key] = np.random.uniform(scanrange[0], scanrange[1], num_models)

    return points


def sample_fixed(parameters: dict) -> dict:
    '''Fixed points given by the config: each parameter carries a per-model list.'''

    points = {}
    for key, pointlist in parameters.items():
        points[key] = np.array(pointlist)

    return points


def sample_grid(parameters: dict, grid_spec: dict) -> tuple:
    """Generate models on a Cartesian grid over specified parameters.

    Parameters listed in ``grid_spec`` (param name → list of explicit grid
    values) are expanded into a Cartesian product; all other parameters are
    sampled uniformly at random (one draw per grid point) from their ranges in
    ``parameters``.

    Returns ``(points, n_grid)`` — the caller overrides num_models with n_grid.
    """
    from itertools import product as cartesian_product

    if not grid_spec:
        raise ValueError("prior='grid' requires a 'grid' section in the YAML config")

    # Build 1-D arrays for each grid parameter
    grid_axes = {}
    for param, values in grid_spec.items():
        grid_axes[param] = np.array(values, dtype=float)
        log.info(f"Grid axis: {param} → {len(values)} points")

    # Cartesian product of all grid axes
    axis_names = list(grid_axes.keys())
    axis_arrays = [grid_axes[n] for n in axis_names]
    grid_points = list(cartesian_product(*axis_arrays))
    n_grid = len(grid_points)
    log.info(f"Grid total: {n_grid} points from {len(axis_names)} axes")

    if n_grid > 10000:
        log.fatal(f"Grid produces {n_grid} points — exceeds the 10,000 model limit! "
                   "Reduce grid axes or split into multiple jobs using genModels.py --num_jobs N.")
        exit()

    points = {}

    # Populate points from the grid
    for i, name in enumerate(axis_names):
        points[name] = np.array([pt[i] for pt in grid_points])

    # Random-sample remaining parameters from their ranges
    for key, scanrange in parameters.items():
        if key in grid_spec:
            continue
        if isinstance(scanrange[0], list):
            # Interrupted range — weighted sub-ranges
            widths = [sr[1] - sr[0] for sr in scanrange]
            total = sum(widths)
            probs = [w / total for w in widths]
            vals = np.empty(n_grid)
            for j in range(n_grid):
                idx = np.random.choice(len(scanrange), p=probs)
                vals[j] = np.random.uniform(scanrange[idx][0], scanrange[idx][1])
            points[key] = vals
        elif scanrange[0] == scanrange[1]:
            # Fixed value
            points[key] = np.full(n_grid, scanrange[0])
        else:
            points[key] = np.random.uniform(scanrange[0], scanrange[1], n_grid)

    log.info(f"Grid sampling complete: {n_grid} models")

    return points, n_grid


def sample_importance(parameters: dict, importance: dict) -> tuple:
    """Generate models by perturbing known viable seed parameters.

    Reads the seed CSV from ``importance['seed_file']``, applies Gaussian
    perturbations with widths from ``importance['proposal_widths']`` (default
    2% of range), and generates perturbations_per_seed variants of each seed.
    Uses propose() for boundary-aware perturbation.

    Returns ``(points, n_total)`` — the caller overrides num_models with n_total.
    """
    import csv

    if not importance:
        raise ValueError("prior='importance' requires an 'importance' section in the YAML config")

    seed_file = importance['seed_file']
    perturbations_per_seed = importance.get('perturbations_per_seed', 10)

    # Read seed CSV
    if not os.path.isfile(seed_file):
        raise FileNotFoundError(f"Seed file not found: {seed_file}")

    with open(seed_file) as f:
        reader = csv.DictReader(f)
        seeds = list(reader)

    n_seeds = len(seeds)
    if n_seeds == 0:
        raise ValueError(f"No seeds found in {seed_file}")

    n_total = n_seeds * perturbations_per_seed
    log.info(f"Importance sampling: {n_seeds} seeds × {perturbations_per_seed} perturbations = {n_total} models")

    if n_total > 10000:
        log.fatal(f"Importance sampling produces {n_total} models — exceeds the 10,000 limit!")
        exit()

    # Build proposal widths: use configured widths, default to 2% of param range
    configured_widths = importance.get('proposal_widths', {})
    param_names = [k for k in parameters.keys()]

    default_widths = {}
    for key, scanrange in parameters.items():
        if isinstance(scanrange[0], list):
            total_range = sum(sr[1] - sr[0] for sr in scanrange)
        elif scanrange[0] == scanrange[1]:
            total_range = 0
        else:
            total_range = scanrange[1] - scanrange[0]
        default_widths[key] = 0.02 * total_range

    widths = {}
    for key in param_names:
        widths[key] = configured_widths.get(key, default_widths[key])
        if widths[key] > 0:
            log.info(f"  {key}: width = {widths[key]:.2f}")

    # Initialize points arrays
    points = {}
    for key in param_names:
        points[key] = np.empty(n_total)

    # Generate perturbations
    idx = 0
    for seed_row in seeds:
        # Build current point from seed CSV
        current = {}
        for key in param_names:
            current[key] = float(seed_row[key])

        for _ in range(perturbations_per_seed):
            proposed = propose(current, widths, parameters)
            for key in param_names:
                points[key][idx] = proposed[key]
            idx += 1

    log.info(f"Importance sampling complete: {n_total} models from {n_seeds} seeds")

    return points, n_total


# ── Proposal helpers (shared by importance sampling and the MH chain) ──────

def build_proposal_widths(parameters: dict, mcmc: dict) -> dict:
    '''Build dict of proposal widths, using configured values or 5% of range as default.'''
    configured = mcmc.get('proposal_widths', {})
    widths = {}
    for key, scanrange in parameters.items():
        if key in configured:
            widths[key] = configured[key]
        else:
            if isinstance(scanrange[0], list):
                total_range = sum(sr[1] - sr[0] for sr in scanrange)
            else:
                total_range = scanrange[1] - scanrange[0]
            widths[key] = abs(total_range) * 0.05
    return widths


def sample_initial_point(parameters: dict) -> dict:
    '''Sample a random starting point within all parameter bounds.'''
    point = {}
    for key, scanrange in parameters.items():
        if isinstance(scanrange[0], list):
            # Pick a random subrange weighted by width, then uniform within it
            range_widths = [sr[1] - sr[0] for sr in scanrange]
            total = sum(range_widths)
            probs = [w / total for w in range_widths]
            idx = np.random.choice(len(scanrange), p=probs)
            point[key] = np.random.uniform(scanrange[idx][0], scanrange[idx][1])
        else:
            point[key] = np.random.uniform(scanrange[0], scanrange[1])
    return point


def propose(current: dict, widths: dict, parameters: dict) -> dict:
    '''Propose a new point using Gaussian perturbation with boundary handling.'''
    proposed = {}
    for key, value in current.items():
        if widths[key] == 0:
            proposed[key] = value
            continue

        new_val = value + np.random.normal(0, widths[key])
        scanrange = parameters[key]

        if isinstance(scanrange[0], list):
            # Interrupted range: reject this dimension if proposed falls in a gap
            in_range = any(sr[0] <= new_val <= sr[1] for sr in scanrange)
            proposed[key] = new_val if in_range else value
        else:
            proposed[key] = reflect(new_val, scanrange[0], scanrange[1])

    return proposed


def reflect(value: float, low: float, high: float) -> float:
    '''Reflect value into [low, high] interval to preserve detailed balance.'''
    if low == high:
        return low
    while value < low or value > high:
        if value < low:
            value = 2 * low - value
        if value > high:
            value = 2 * high - value
    return value
