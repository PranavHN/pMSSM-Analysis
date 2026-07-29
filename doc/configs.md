# Configuration System

This document describes the configuration system used by Run3ModelGen, including the config file structure, the available priors, and the `genConfig.py` helper script.

## Config file structure

All scan parameters are controlled via YAML config files. When `genModels.py` is run, any missing keys are filled from the [default config](../source/Run3ModelGen/data/configs/default_config.yaml).

### Core fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prior` | string | `flat` | Sampling method: `flat`, `fixed`, `mcmc`, `grid`, or `importance`. |
| `num_models` | int | `20` | Number of models to generate (max 10,000 per job). |
| `isGMSB` | bool | `False` | Enable GMSB-specific parameter handling. |
| `parameters` | dict | See default | pMSSM parameter ranges (flat/mcmc) or fixed values (fixed). |
| `steps` | list | See default | Pipeline steps to run, in order. |
| `mcmc` | dict | See default | MCMC-specific settings. Only used when `prior: mcmc`. See [doc/mcmc.md](mcmc.md). |
| `constraints` | list | `[]` | Physics constraints for MCMC likelihood. See [doc/constraints.md](constraints.md). |
| `grid` | dict | `{}` | Grid axis definitions. Only used when `prior: grid`. Maps parameter names to lists of values. |
| `importance` | dict | `{}` | Importance sampling settings. Only used when `prior: importance`. See [Importance](#importance) section. |

### Parameter specification

The format of the `parameters` block depends on the prior:

**Flat prior** - each parameter is a `[min, max]` range:
```yaml
parameters:
  tanb: [1, 60]
  M_1: [-2000, 2000]
```

**Fixed prior** - each parameter is a list of specific values (one per model):
```yaml
parameters:
  tanb: [10, 20, 30]
  M_1: [100, 200, 300]
```

**MCMC prior** - same format as flat (ranges):
```yaml
parameters:
  tanb: [1, 60]
  M_1: [-2000, 2000]
```

**Grid prior** - same format as flat (ranges) for non-grid parameters. Parameters that appear in the `grid` block take their values from the Cartesian product; all others are drawn uniformly at random from their `parameters` ranges:
```yaml
parameters:
  tanb: [5, 55]        # randomised (not on grid)
  M_1: [-550, 550]     # grid axis (overridden by grid block)
  M_2: [2000, 2000]    # fixed (min == max)

grid:
  M_1: [-500, -200, 200, 500]
  tanb: [10, 30, 50]
```

**Importance prior** - same format as flat (ranges) for boundary enforcement. Actual initial values come from the seed CSV:
```yaml
parameters:
  tanb: [1, 60]
  M_1: [-2000, 2000]

importance:
  seed_file: seeds.csv
  perturbations_per_seed: 10
  proposal_widths:
    M_1: 10.0
    tanb: 2.0
```

**Gaps in ranges** are supported for flat, mcmc, and importance priors by providing subranges:
```yaml
parameters:
  M_2: [[-3000, -1000], [1000, 3000]]
```

### Pipeline steps

The `steps` list defines which physics tools to run and in what order. Each step has:

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | Step name, must match a `run_<name>` method (e.g. `SPheno`, `softsusy`). |
| `output_dir` | Yes | Directory for output files (relative to scan dir). |
| `prefix` | Yes | Prefix for ntuple variable names (e.g. `SP` for SPheno). |
| `input_dir` | No | Directory to read input from (relative to scan dir). |
| `log_dir` | No | Directory for log files (relative to scan dir). |

Available steps: `prep_input`, `SPheno`, `softsusy`, `micromegas`, `superiso`, `gm2calc`, `feynhiggs`, `SModelS`.

## Priors

### Flat

Draws parameters uniformly at random within the specified ranges. The pipeline runs after all models are sampled.

```yaml
prior: flat
num_models: 1000
```

### Fixed

Runs the pipeline on a specific set of parameter points. The number of models is inferred from the length of the parameter lists.

```yaml
prior: fixed
parameters:
  tanb: [10, 20, 30]   # 3 models
  M_1: [100, 200, 300]
```

### MCMC

Uses Metropolis-Hastings sampling guided by physics constraints. The pipeline runs for each proposal during sampling. Requires an `mcmc` block and a `constraints` list. See [doc/mcmc.md](mcmc.md) for full details.

```yaml
prior: mcmc
num_models: 500
mcmc:
  burn_in: 100
  max_attempts: 50000
  accept_invalid: false
constraints:
  - name: higgs_mass
    ...
```

### Grid

Generates models on a Cartesian product over explicitly specified parameter values. This is useful for systematically covering a parameter subspace, for instance when targeting narrow phenomenological corridors that MCMC struggles to explore.

The `grid` block maps parameter names to lists of values. The total number of models equals the product of the list lengths, overriding `num_models`. Parameters **not** listed in `grid` are drawn uniformly at random from their `parameters` ranges (one independent draw per grid point). Fixed parameters (`[val, val]`) remain constant. Interrupted ranges are supported for non-grid parameters.

```yaml
prior: grid
num_models: 600  # overridden by grid product (8 × 5 × 5 × 3 = 600)

parameters:
  tanb: [5, 55]
  M_1: [-550, 550]
  M_2: [2000, 2000]
  AT: [-6000, 6000]
  meL: [150, 650]
  meR: [150, 650]

grid:
  M_1: [-500, -350, -200, -100, 100, 200, 350, 500]   # 8 values
  meL: [160, 280, 400, 520, 640]                        # 5 values
  meR: [160, 280, 400, 520, 640]                        # 5 values
  tanb: [10, 30, 50]                                    # 3 values
```

Key differences from other priors:
- The `num_models` value is **overridden** by the total grid size.
- The 10,000-model-per-job cap applies: if the grid product exceeds 10,000, split into multiple jobs (`genModels.py --num_jobs N`).
- Like `flat` and `fixed`, the pipeline runs after all models are sampled (unlike `mcmc`, which runs the pipeline during sampling).

### Importance

Generates models by perturbing known viable seed parameters with narrow Gaussian widths. This is designed for exploring narrow corridors in parameter space where MCMC and grid sampling both fail (e.g. co-annihilation strips where the viable region is ~10-20 GeV wide).

The `importance` block specifies:

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `seed_file` | Yes | — | Path to CSV file with seed parameters (columns must match parameter names). |
| `perturbations_per_seed` | No | `10` | Number of Gaussian perturbations per seed row. |
| `proposal_widths` | No | 2% of range | Per-parameter Gaussian step sizes. |

```yaml
prior: importance
num_models: 620  # overridden by n_seeds × perturbations_per_seed

importance:
  seed_file: results/seeds.csv
  perturbations_per_seed: 10
  proposal_widths:
    M_1: 10.0       # narrow — preserve LSP identity
    meL: 10.0       # narrow — co-annihilation strip
    AT: 200.0        # moderate — affects Higgs mass

parameters:
  tanb: [5, 60]      # boundary enforcement
  M_1: [-600, 600]
  # ...
```

The seed CSV must have a header row with columns matching the parameter names in the `parameters` block. Additional columns (e.g. `scan_dir`, `model_id`) are ignored. Each seed row is perturbed `perturbations_per_seed` times using Gaussian noise, with boundary reflection via `_propose()` / `_reflect()` (the same methods used by MCMC).

Key differences from other priors:
- The `num_models` value is **overridden** by `n_seeds × perturbations_per_seed`.
- Parameters without a `proposal_widths` entry default to 2% of their total range. Set to `0` to keep a parameter fixed at its seed value.
- The 10,000-model-per-job cap applies.
- Like `flat` and `grid`, the pipeline runs after all models are sampled.

## Config file location

Config files are stored in `source/Run3ModelGen/data/configs/`:

| File | Description |
| --- | --- |
| `default_config.yaml` | Default values for all config keys. Read automatically. |
| `example_flat_config.yaml` | Example flat prior config with full pipeline. |
| `example_fixed_config.yaml` | Example fixed prior config with 3 parameter points. |
| `example_mcmc_config.yaml` | Example MCMC config with 4 physics constraints. |
| `example_grid_config.yaml` | Example grid prior config with 3 grid axes. |
| `example_importance_config.yaml` | Example importance sampling config with seed CSV. |

## genConfig.py

The `genConfig.py` script copies an example config file to a specified directory:

```bash
genConfig.py flat                      # Copies example_flat_config.yaml to current dir
genConfig.py mcmc --output_dir myrun/  # Copies example_mcmc_config.yaml to myrun/
genConfig.py fixed                     # Copies example_fixed_config.yaml to current dir
genConfig.py grid                      # Copies example_grid_config.yaml to current dir
genConfig.py importance                # Copies example_importance_config.yaml to current dir
```

### Usage

```
genConfig.py PRIOR [--output_dir DIR]

Arguments:
  PRIOR       The prior type: flat, fixed, mcmc, grid, or importance

Options:
  --output_dir  Directory to save the config file (default: current directory)
```
