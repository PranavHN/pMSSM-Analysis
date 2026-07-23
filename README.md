# Run3ModelGen <!-- omit in toc -->

Draft repo for updated shiny pMSSM model generation. Aiming for this repo to be fully independent of HTCondor & lxplus as well as being fully Dockerised.

# Table of Contents <!-- omit in toc -->

- [Installation](#installation)
  - [pixi](#pixi)
  - [Run3ModelGen](#run3modelgen)
  - [Environments](#environments)
- [On startup](#on-startup)
- [Running](#running)
  - [Sampling priors](#sampling-priors)
  - [Changing the scan configuration](#changing-the-scan-configuration)
  - [MCMC sampling](#mcmc-sampling)
    - [Diagnostics](#diagnostics)
  - [Grid sampling](#grid-sampling)
  - [Importance sampling](#importance-sampling)
  - [Extracting interesting models](#extracting-interesting-models)
    - [From scan directories](#from-scan-directories)
    - [From .tar balls directly](#from-tar-balls-directly)
  - [HTCondor](#htcondor)
    - [Generating models](#generating-models)
    - [Extracting models](#extracting-models)
- [Documentation](#documentation)
- [Reference files \& CI regression check](#reference-files--ci-regression-check)
- [FAQ](#faq)
  - [Adding software/updating the software version](#adding-softwareupdating-the-software-version)
  - [Fixing the parameter points](#fixing-the-parameter-points)
  - [Gaps in the scan ranges](#gaps-in-the-scan-ranges)
  - [cmake\_build\_env.sh](#cmake_build_envsh)
  - [pixi](#pixi-1)

## Software content

| Software | Implemented Version | Notes |
| - | - | - |
| SPheno | 4.0.6 | SUSY spectrum generator with precise mass calculations |
| Softsusy | 4.1.22 | Alternative SUSY spectrum generator for cross-checks |
| FeynHiggs | 2.19.0 | Precise Higgs mass calculation with higher-order corrections |
| SuperIso | 5.0 | Flavor physics observables (B-physics, rare decays) |
| GM2Calc | 2.3.1 | Muon anomalous magnetic moment (g-2) calculation |
| micrOMEGAs | 5.2.1 | Dark matter relic density and direct/indirect detection. Upgrading to version 6.3.0 requires re-generating patch files. See: https://gitlab.cern.ch/jwuerzin/Run3ModelGen/-/issues/15 |
| SModelS | 3.1.1 | LHC SUSY-search reinterpretation / cross-section upper limits. Uses a bundled pythia8 for LO cross sections (built automatically, see [doc/pythia8.md](doc/pythia8.md)) |
| EVADE | - | Direct detection constraints (currently unavailable). Version 0.2.0 broke after a recent update to HOM4PS and will remain unavailable for the time being. |

# Installation

## pixi

This repo uses [pixi](https://pixi.sh/latest/) to manage packages, compilers, etc. Hopefully this will only make things easier for you as a user, but it does require installing pixi locally before we can compile the Run3ModelGen code. To install pixi, do:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bash_profile
```

or if you are working on a CVMFS-enabled machine you can setup the latest Pixi release with

```
. /cvmfs/cms-griddata.cern.ch/cat/sw/pixi/latest/setup.sh
```

Please note that since pixi edits your `.bash_profile` automatically, it should be set up correctly on your next login.

## Run3ModelGen

Having installed pixi, we can use the `pixi.toml` and `pixi.lock` files to start a shell with the correct environment and install the Run3ModelGen:

```bash
pixi shell
cmake -S source -B build
cmake --build build -j8
source build/setup.sh
```

## Environments

The project provides two pixi environments:

**Default environment** — contains the full build toolchain (compilers, cmake), all Python dependencies, and the external physics software. This is what you use for building and running scans. ROOT ntuples are produced and read using [uproot](https://uproot.readthedocs.io/), a pure-Python library for ROOT file I/O that does **not** require a ROOT installation. uproot is included in the default environment automatically.

**`rootenv` environment** — a lightweight, standalone environment containing only Python 3.11 and [ROOT](https://root.cern/) (>= 6.40.0, via `root_base`). It does **not** include the build toolchain or any physics software. Use this for tasks that require the full ROOT framework (TBrowser, RDataFrame, PyROOT, ROOT macros, etc.):

```bash
pixi shell -e rootenv                    # Enter the ROOT environment
pixi run -e rootenv python my_analysis.py  # Run a script in rootenv
```

Since `rootenv` is defined with `no-default-feature = true`, it is fully independent of the default environment and resolves quickly. For more details on both environments, see [doc/pixi.md](doc/pixi.md).

# On startup

As perviously mentioned, pixi should be ready to use as per your `.bash_profile`, so there is no need to do anything here.
For settup up the Run3ModelGen, please simply execute:

```bash
pixi shell
source build/setup.sh
```
And you're good to go!

If you need ROOT for analysis, use the `rootenv` environment instead:
```bash
pixi shell -e rootenv
```

# Running

The main script for generating models is [genModels.py](source/Run3ModelGen/scripts/genModels.py). If you set everything up correctly, this can simply be executed with:

```bash
genModels.py
```

The script will save the created model files and the corresponding root ntuple.

## Sampling priors

Run3ModelGen supports five sampling priors:

| Prior | Description |
| --- | --- |
| `flat` | Draws parameters uniformly at random within specified ranges. |
| `fixed` | Runs the pipeline on a specific set of parameter points. |
| `mcmc` | Metropolis-Hastings MCMC guided by physics constraints. See [MCMC sampling](#mcmc-sampling). |
| `grid` | Cartesian product over specified parameter values. See [Grid sampling](#grid-sampling). |
| `importance` | Perturbs known viable seed models with narrow Gaussians. See [Importance sampling](#importance-sampling). |

The prior is set in the config file via `prior: flat`, `prior: fixed`, `prior: mcmc`, `prior: grid`, or `prior: importance`.

## Changing the scan configuration

If you would like to change the scan configuration or seed, simply run

```bash
genModels.py --config_file myconfig.yaml --seed 42
```

where `myconfig.yaml` is your config file containing the scan configuration. If no config file is supplied, the [default config file](source/Run3ModelGen/data/configs/default_config.yaml) will be read. All values that are missing from `myconfig.yaml` are read from the default as well. For full details on the config file format, see [doc/configs.md](doc/configs.md).

To generate an example config file for a given prior, use the `genConfig.py` script:

```bash
genConfig.py flat                      # Generates example_flat_config.yaml in current dir
genConfig.py mcmc --output_dir myrun/  # Generates example_mcmc_config.yaml in myrun/
genConfig.py fixed                     # Generates example_fixed_config.yaml in current dir
genConfig.py grid                      # Generates example_grid_config.yaml in current dir
genConfig.py importance                # Generates example_importance_config.yaml in current dir
```

Example configs are stored in [source/Run3ModelGen/data/configs/](source/Run3ModelGen/data/configs/).

## MCMC sampling

MCMC sampling uses the Metropolis-Hastings algorithm to explore parameter space guided by physics observables. Instead of sampling uniformly, the chain preferentially visits regions where computed observables (e.g. Higgs mass, relic density) agree with experimental measurements.

To use MCMC, set `prior: mcmc` and define constraints in your config file:

```yaml
prior: mcmc
num_models: 500

mcmc:
  burn_in: 100
  max_attempts: 50000
  accept_invalid: false

constraints:
  - name: higgs_mass
    source:
      step: SPheno
      file_type: slha
      block: MASS
      entry: 25
    type: gaussian
    central: 125.09
    uncertainty: 3.0
```

Key MCMC settings:
- **`burn_in`**: Number of accepted steps to discard before collecting models.
- **`max_attempts`**: Hard cap on total proposals (includes burn-in).
- **`accept_invalid`**: If `false` (default), models where the pipeline failed are not saved.
- **`proposal_widths`**: Optional per-parameter Gaussian step sizes (default: 5% of range).
- **`adapt_proposals`** (optional): Robbins-Monro adaptive scaling of `proposal_widths` *during burn-in only*, to drive the rolling Metropolis acceptance rate toward `target_acceptance` (default 0.25 — Roberts-Gelman-Gilks high-D optimum). Adaptation stops automatically when burn-in ends so the post-burn-in chain remains detailed-balance-correct. Recommended unless you've already hand-tuned widths.
  ```yaml
  mcmc:
    adapt_proposals:
      enabled: true
      target_acceptance: 0.25
      window: 200
      learning_rate: 0.05
  ```

### Diagnostics

Each chain emits per-window log lines with the *Metropolis acceptance rate* (`accept_metro`) and the pipeline-validity rate, plus a parseable end-of-run summary line:

```
[mcmc-summary] proposed=12345 metro_accepted=3094 metro_accept_rate=0.2506 ...
```

Use `accept_metro` (not the commit rate) to judge proposal-width tuning — values outside ~10–60 % trigger a warning. To check across-chain convergence after a batch of jobs, run:

```bash
python source/Run3ModelGen/scripts/mcmc_diagnostics.py --data-dir /path/to/ntuple_dir
# Or skip the file-level Ω-straddle filter to diagnose the unfiltered sampler:
python source/Run3ModelGen/scripts/mcmc_diagnostics.py --data-dir /path/to/ntuple_dir --no-file-cut
```

This computes split-R̂ (Vehtari et al. 2021) and bulk-ESS per free parameter across all chain ROOT files in the directory, treating each file as a separate chain. Targets are `R̂ < 1.05` and `ESS > 400`. R̂ ≫ 1 means chains haven't mixed (likely needs wider proposals or longer chains); ESS ≪ raw sample count means within-chain autocorrelation is long (likely needs wider proposals).

For a tuned starting config on relic-density-only scans, see [`example_mcmc_relic_tuned.yaml`](source/Run3ModelGen/data/configs/example_mcmc_relic_tuned.yaml).

For full details, see [doc/mcmc.md](doc/mcmc.md). For constraint specification, see [doc/constraints.md](doc/constraints.md).

## Grid sampling

Grid sampling generates models on a Cartesian product over explicitly specified parameter values. This is useful for systematically covering a parameter subspace, for instance when targeting narrow phenomenological corridors that MCMC struggles to explore.

To use grid sampling, set `prior: grid` and add a `grid` block mapping parameter names to lists of values:

```yaml
prior: grid
num_models: 600  # overridden by the grid product

parameters:
  tanb: [5, 55]             # randomised (not on grid)
  M_1: [-550, 550]          # grid axis (overridden below)
  M_2: [2000, 2000]         # fixed
  AT: [-6000, 6000]         # randomised
  meL: [150, 650]           # grid axis (overridden below)
  meR: [150, 650]           # grid axis (overridden below)
  # ... other parameters ...

grid:
  M_1: [-500, -350, -200, -100, 100, 200, 350, 500]   # 8 values
  meL: [160, 280, 400, 520, 640]                        # 5 values
  meR: [160, 280, 400, 520, 640]                        # 5 values
  tanb: [10, 30, 50]                                    # 3 values
```

This produces 8 x 5 x 5 x 3 = 600 grid points. Parameters listed in the `grid` block take their values from the Cartesian product; all other parameters are drawn uniformly at random (one independent draw per grid point) from their `parameters` ranges. Fixed parameters (`[val, val]`) remain constant across all grid points. Interrupted ranges are also supported for non-grid parameters.

Key differences from other priors:
- The `num_models` value in the config is **overridden** by the total grid size.
- The 10,000-model-per-job cap applies: if the grid product exceeds 10,000, split into multiple jobs.
- Like `flat` and `fixed`, the pipeline runs after all models are sampled (unlike `mcmc`, which runs the pipeline during sampling).

For full details on the config format, see [doc/configs.md](doc/configs.md).

## Importance sampling

Importance sampling generates models by perturbing known viable parameter points with narrow Gaussian widths. This is useful for exploring narrow corridors in parameter space (e.g. co-annihilation strips, compressed spectra) where both MCMC and grid sampling fail because the viable region is too narrow to find from random or evenly-spaced starting points.

To use importance sampling, set `prior: importance` and provide a seed CSV file plus perturbation settings:

```yaml
prior: importance
num_models: 100  # overridden by n_seeds × perturbations_per_seed

importance:
  seed_file: seeds.csv          # CSV with one row per seed model
  perturbations_per_seed: 10    # Gaussian perturbations per seed
  proposal_widths:              # per-parameter step sizes (optional)
    M_1: 10.0                   # narrow — must stay near seed
    meL: 10.0
    AT: 200.0
```

The seed CSV must have a header row with columns matching the pMSSM parameter names in the config (e.g. `M_1`, `M_2`, `tanb`, ...). Additional columns (`scan_dir`, `model_id`) are ignored. Each seed is perturbed `perturbations_per_seed` times using Gaussian noise with the specified widths, reflected at the boundaries defined in `parameters`.

Key points:
- The `num_models` value is **overridden** by `n_seeds × perturbations_per_seed`.
- Parameters without an explicit `proposal_widths` entry default to 2% of their range.
- Uses the same `_propose()` / `_reflect()` methods as MCMC for boundary-aware perturbation.
- Like `flat` and `grid`, the pipeline runs after all models are sampled.

For full details on the config format, see [doc/configs.md](doc/configs.md).

## Extracting interesting models

Interesting models to be fed into the [pMSSMFactory](https://gitlab.cern.ch/atlas-phys-susy-wg/summaries/pmssm/pMSSMFactory) can be extracted using [extractModels.py](source/Run3ModelGen/scripts/extractModels.py). This script works both with directories and .tar balls.

### From scan directories

Interesting models can be extracted from a scan directory with e.g.
```bash
extractModels.py --scan_dir scan --root_file scan/ntuple.0.0.root --selection "(akarr['SS_m_h']!=-1) & (akarr['SS_m_h'] <= 130)"
```

where `--scan_dir` and `--root_file` are required arguments. The resulting set of models can be read into the [pMSSMFactory](https://gitlab.cern.ch/atlas-phys-susy-wg/summaries/pmssm/pMSSMFactory) directly and is stored in `$EOSPATH/SelectedModels/<scan_dir>/Models/<subdir>/<ProcId>`.

Here, `<scan_dir>` is the argument supplied to `--scan_dir` (here: `scan`), `<subdir>` the sub-directory for the respective model generation step (e.g. `softsusy`) and `<ProcId>` the number indicated in the NTuple name before the suffix, i.e. `ntuple.XYZ.<ProcId>.root` (here: `0`).

Please note that as of now, the selection has to be supplied in a format that can be used to mask awkward arrays. Feel free to let me know if this causes any issues for you!

### From .tar balls directly

`extractModels.py` can take `.tar` balls as input, so there is no need to un-tar a scan directory before running. This can be done by calling e.g.
```bash
extractModels.py --scan_dir /eos/user/j/jwuerzin/Run3ModelGen/15280231/0/scan.15280231.0.tar.gz --root_file /eos/user/j/jwuerzin/Run3ModelGen/15280231/0/ntuple.15280231.0.root --selection "akarr['SP_m_h'] < 125."
```

## HTCondor

### Generating models

For larger scans, parallel jobs can be submitted with [genModels_condor.py](source/Run3ModelGen/scripts/genModels_condor.py):
```bash
genModels_condor.py --config_file myconfig.yaml --condor_dir condorlogdir --num_jobs 42
```

This will automatically submit `num_jobs` to the HTCondor batch system, while storing the relevant outputs in `condorlogdir`. Please note that since each config file contains `num_models` models, this will overall result in `num_jobs` x `num_models` attempted models.

The output .tar ball containing all models and the corresponding ntuple will automatically be stored in `$EOSPATH`, which is automatically defined during compilation in `build/setup.sh`.

The HTCondor job flavour can be changed using the `--flavour` flag.

### Extracting models

Models produced with HTCondor can also be extracted with batch jobs. To do so, call [extractModels_condor.py](source/Run3ModelGen/scripts/extractModels_condor.py) with e.g.:
```bash
extractModels_condor.py --eos_dir /eos/user/j/jwuerzin/Run3ModelGen/15280231 --selection "akarr['SP_m_h'] < 125."
```

The output `Models` directory can be fed into the pMSSMFactory directly and is stored in `$EOSPATH/SelectedModels/<ClusterId>/`. The corresponding config .yaml file is also stored there.

The HTCondor job flavour can be changed using the `--flavour` flag.

# Documentation

Detailed documentation is available in the [doc/](doc/) directory:

- [doc/ntuple.md](doc/ntuple.md) - NTuple structure: tree layout, branch naming, and per-tool branch groups
- [doc/mcmc.md](doc/mcmc.md) - MCMC sampling algorithm, configuration, and output
- [doc/constraints.md](doc/constraints.md) - Physics constraints specification (types, sources, examples)
- [doc/configs.md](doc/configs.md) - Configuration system, prior types, and `genConfig.py` usage
- [doc/pixi.md](doc/pixi.md) - pixi package manager: environments, tasks, dependencies
- [doc/micromegas_build.md](doc/micromegas_build.md) - micrOMEGAs build notes
- [doc/pythia8.md](doc/pythia8.md) - SModelS pythia8 auto-build (workaround for broken upstream download)

# Reference files & CI regression check

Each CI `genModels` job compares the **per-step success counts** of its scan against a
reference stored on EOS and fails on a mismatch. The references are (re)generated with two
**manual** pixi tasks (never run automatically in CI) — produce first, review, then upload:

```bash
# 1. produce the reference scans and print their per-step summary (no upload, no credentials)
pixi run make-refs

# 2. once you're happy with the summary, upload the references to EOS
export CERN_USER=nicebot                                                # the service account
export SERVICE_PASS="$(printf %s 'THE_NICEBOT_PASSWORD' | base64 -w0)"  # BASE64 of the password (never commit it)
pixi run upload-refs
```

`make-refs` runs all 5 CI scans and prints their per-step success counts, leaving the ntuples
under `run/scan_<name>/` (no credentials needed). `upload-refs` then uploads, per scan,
`ntuple_<scan>.root` and a `counts_<scan>.json` sidecar to
`root://eosuser.cern.ch//eos/user/n/nicebot/Run3ModelGen/` (the CI jobs download the sidecar
via `compare_reference.py` and compare). It reads the credentials from the environment,
`kinit`s into a private credential cache destroyed on exit (so it neither prints the password
nor overwrites your own ticket), and errors clearly if a reference is missing or the variables
are unset/invalid. Both tasks accept `--scans flat,mcmc` to act on a subset.

**Prerequisites for the CI check to pass:** set the masked CI variables `SERVICE_PASS`
(base64 of the `nicebot` password) and `CERN_USER` (`nicebot`), and run `make-refs` then
`upload-refs` once to seed the references. Until then the comparison jobs fail with a clear
"reference not found" message.

# FAQ

More will follow soon. In the meantime, if you have any questions at all, please contact jonas.wuerzinger@cern.ch.

## Adding software/updating the software version

Most external tools are provided as [conda-forge](https://conda-forge.org/) packages; to change a version, edit its constraint in `pixi.toml` (see [doc/pixi.md](doc/pixi.md)). The only tool still built from source in this cmake workflow is micrOMEGAs — see:
- [doc/micromegas_build](doc/micromegas_build.md)

SModelS's bundled pythia8 is also built automatically at build time (working around a broken upstream download); see [doc/pythia8.md](doc/pythia8.md).

## Fixing the parameter points

To generate models with fixed parameters, simply supply a config file with a `fixed` prior, while supplying the parameters in a list. The example below will generate two models:
```yaml
...
prior: fixed
parameters:
  tanb: [10, 60]
  M_1: [-1000, 2000]
  M_2: [-111, 20]
...
```

## Gaps in the scan ranges

If you'd like to run with gaps in the scan ranges, this is fully supported. Simply provide the scan ranges as a list containing the subranges, e.g.:
```yaml
...
parameters:
  tanb: [1, 60]
  M_1: [-2000, 2000]
  M_2: [[-3000, -1000], [-1000, 800], [1000, 3000]]
...
```

## cmake_build_env.sh

The [cmake_build_env.sh](source/cmake_build_env.sh) script is automatically called during the CMake build process and sets up the build environment. You typically don't need to run this script manually.

**What it does:**
- Creates the necessary build directory structure
- Sets up symbolic links for Python modules from `source/Run3ModelGen/python/` to `build/Run3ModelGen/`
- Creates the EOS output directory at `/eos/user/<initial>/<username>/Run3ModelGen`
- Generates the `build/setup.sh` file with all necessary environment variables:
  - `PATH`: Adds the micrOMEGAs binary and the Run3ModelGen scripts (the other physics tools come from their conda-forge packages in the pixi environment)
  - `PYTHONPATH`: Adds the build directory for Python module imports
  - `DATAPATH`: Points to the Run3ModelGen data directory
  - `EOSPATH`: Points to your EOS output directory
  - `RUN3MODELGENDIR`: Points to the repository root directory

**When it's called:**
This script is executed automatically by CMake during the build process (`cmake --build build -j8`). The `-s` (source) and `-b` (build) directory arguments are provided by the CMake configuration.

**Manual usage (advanced):**
If you need to run it manually for debugging purposes:
```bash
source/cmake_build_env.sh -s source -b build
```

## pixi

For detailed information on pixi usage, environments (including the lightweight `rootenv`), tasks, dependency management, and troubleshooting, see [doc/pixi.md](doc/pixi.md).

**Important:** Never edit `pixi.lock` directly — it is generated automatically from `pixi.toml`.
