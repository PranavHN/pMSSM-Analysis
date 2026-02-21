# Configuration System

This document describes the configuration system used by Run3ModelGen, including the config file structure, the available priors, and the `genConfig.py` helper script.

## Config file structure

All scan parameters are controlled via YAML config files. When `genModels.py` is run, any missing keys are filled from the [default config](../source/Run3ModelGen/data/configs/default_config.yaml).

### Core fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prior` | string | `flat` | Sampling method: `flat`, `fixed`, or `mcmc`. |
| `num_models` | int | `20` | Number of models to generate (max 10,000 per job). |
| `isGMSB` | bool | `False` | Enable GMSB-specific parameter handling. |
| `parameters` | dict | See default | pMSSM parameter ranges (flat/mcmc) or fixed values (fixed). |
| `steps` | list | See default | Pipeline steps to run, in order. |
| `mcmc` | dict | See default | MCMC-specific settings. Only used when `prior: mcmc`. See [doc/mcmc.md](mcmc.md). |
| `constraints` | list | `[]` | Physics constraints for MCMC likelihood. See [doc/constraints.md](constraints.md). |

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

**Gaps in ranges** are supported for flat and mcmc priors by providing subranges:
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

## Config file location

Config files are stored in `source/Run3ModelGen/data/configs/`:

| File | Description |
| --- | --- |
| `default_config.yaml` | Default values for all config keys. Read automatically. |
| `example_flat_config.yaml` | Example flat prior config with full pipeline. |
| `example_fixed_config.yaml` | Example fixed prior config with 3 parameter points. |
| `example_mcmc_config.yaml` | Example MCMC config with 4 physics constraints. |

## genConfig.py

The `genConfig.py` script copies an example config file to a specified directory:

```bash
genConfig.py flat                      # Copies example_flat_config.yaml to current dir
genConfig.py mcmc --output_dir myrun/  # Copies example_mcmc_config.yaml to myrun/
genConfig.py fixed                     # Copies example_fixed_config.yaml to current dir
```

### Usage

```
genConfig.py PRIOR [--output_dir DIR]

Arguments:
  PRIOR       The prior type: flat, fixed, or mcmc

Options:
  --output_dir  Directory to save the config file (default: current directory)
```
