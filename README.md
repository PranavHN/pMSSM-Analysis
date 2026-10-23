# Run3ModelGen <!-- omit in toc -->

Modern pMSSM model generation, orchestrated by a [Snakemake](https://snakemake.readthedocs.io) workflow: the same entrypoints run locally, on SLURM, or on HTCondor (`--backend`), with resumable scans and per-step re-runs. Fully independent of lxplus, fully Dockerised.

# Table of Contents <!-- omit in toc -->

- [Installation](#installation)
  - [pixi](#pixi)
  - [Run3ModelGen](#run3modelgen)
  - [Environments](#environments)
- [On startup](#on-startup)
- [Running](#running)
  - [Commands](#commands)
  - [Sampling priors](#sampling-priors)
  - [Changing the scan configuration](#changing-the-scan-configuration)
  - [MCMC sampling](#mcmc-sampling)
    - [Diagnostics](#diagnostics)
  - [Grid sampling](#grid-sampling)
  - [Importance sampling](#importance-sampling)
  - [Active learning](#active-learning)
  - [Extracting interesting models](#extracting-interesting-models)
    - [From scan directories](#from-scan-directories)
    - [From .tar balls directly](#from-tar-balls-directly)
  - [Batch running (SLURM / HTCondor)](#batch-running-slurm--htcondor)
    - [Extracting models from a published run](#extracting-models-from-a-published-run)
- [Documentation](#documentation)
- [Reference files \& CI regression check](#reference-files--ci-regression-check)
- [FAQ](#faq)
  - [Adding software/updating the software version](#adding-softwareupdating-the-software-version)
  - [Fixing the parameter points](#fixing-the-parameter-points)
  - [Gaps in the scan ranges](#gaps-in-the-scan-ranges)
  - [Unlocking an interrupted scan](#unlocking-an-interrupted-scan)
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

Having installed pixi, we can use the `pixi.toml` and `pixi.lock` files to start a shell with the correct environment and install the Run3ModelGen. Install the environments first, then build:

```bash
pixi install -a           # all environments (default + rootenv + dagviz)
pixi run build            # cmake configure + build
pixi shell
source build/setup.sh
```

The two steps are separate on purpose: `pixi install` cannot run from inside a `pixi run` task, so the `build` task only builds. Installing only the default environment (plain `pixi install`) is enough to build and run scans — `-a` additionally gives you [`rootenv`](#environments) and the `dagviz` env that renders each run's `dag.svg`.

or run the build commands yourself, which is the same thing spelled out:

```bash
pixi install -a
pixi shell
cmake -S source -B build
cmake --build build -j8
source build/setup.sh
```

## Environments

The project provides three pixi environments for the standard scans, plus two GPU
environments (`rocm`, `cuda`) that only `prior: active_learning` needs (see
[Active learning](#active-learning) and [doc/pixi.md](doc/pixi.md)):

**Default environment** — contains the full build toolchain (compilers, cmake), all Python dependencies, and the external physics software. This is what you use for building and running scans. ROOT ntuples are produced and read using [uproot](https://uproot.readthedocs.io/), a pure-Python library for ROOT file I/O that does **not** require a ROOT installation. uproot is included in the default environment automatically.

**`rootenv` environment** — a lightweight, standalone environment containing only Python 3.11 and [ROOT](https://root.cern/) (>= 6.40.0, via `root_base`). It does **not** include the build toolchain or any physics software. Use this for tasks that require the full ROOT framework (TBrowser, RDataFrame, PyROOT, ROOT macros, etc.):

```bash
pixi shell -e rootenv                    # Enter the ROOT environment
pixi run -e rootenv python my_analysis.py  # Run a script in rootenv
```

**`dagviz` environment** — [graphviz](https://graphviz.org/) only, used to render the DAG snapshot every run writes. `pixi install -a` installs it along with everything else; on its own:

```bash
pixi install -e dagviz
```

Nothing in it touches xrootd; it is separate because graphviz and xrootd cannot *share* one environment. graphviz pulls GTK3/pango, which are still built against the old `libxml2` (`<2.14`), while xrootd 6.x needs `libxml2-16 >=2.14.6` — and an environment holds only one `libxml2`. Adding graphviz to the default environment therefore downgrades xrootd 6.1.0 → 5.8.4 (measured, plus `libcurl` and `krb5`, and 40 extra packages on every worker node), and xrootd is what `--publish` uses to reach EOS. With `dagviz` installed, every run renders its own `dag.svg` alongside the `dag.dot` source.

Both `rootenv` and `dagviz` are defined with `no-default-feature = true`, so they are fully independent of the default environment and resolve quickly. For more details on the environments, see [doc/pixi.md](doc/pixi.md).

### Optional: detach the environments from the project directory

By default pixi materialises each environment under `.pixi/envs` inside the project. Conda-style environments consist of very many small files, so if the project lives on a filesystem with a file-count (inode) quota, the environments are usually what exhausts it — long before disk space runs out. The symptom is easy to misread: plenty of free space, but every write fails with `Quota exceeded`.

`detached-environments` moves them elsewhere. It is a *global* setting, written to `~/.pixi/config.toml`, so one command covers every project on the machine:

```bash
pixi config set detached-environments /path/to/roomy/filesystem/pixi-envs
pixi install
pixi run python -c 'import sys; print(sys.executable)'
```

The interpreter should now be under the new location. `pixi install` returning without re-solving confirms the relocated environment satisfies `pixi.lock`, so it is the same environment as before.

Then reclaim the old copy and let pixi restore the path:

```bash
rm -rf .pixi/envs && pixi install
ls -ld .pixi/envs                  # expect a symlink to the new location
```

pixi recreates `.pixi/envs` as a **symlink**, which is what makes this safe to retrofit into an existing setup: scripts that hardcode `.pixi/envs/default/bin/python` keep working through the link.

Two caveats. If the new location is a scratch filesystem, check its purge policy first — an environment left untouched long enough can be deleted, needing `pixi install` again, and once several projects point at the same place one purge takes out all of them. On a cluster, the new location must also be readable from the batch nodes, not just the login node. Finally, do not run the `rm -rf` step while a scan is running: model generation shells out through `pixi run` on every iteration, so the environment is in use for the whole job even though the submit scripts never mention pixi.

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

The script will save the created model files and the corresponding root ntuple in the scan directory. It is the single entrypoint for every execution backend — locally or submitted to a cluster — via `--backend`:

```bash
genModels.py --config_file myconfig.yaml --num_jobs 50 --backend slurm     # or htcondor / local
```

See [Batch running](#batch-running-slurm--htcondor) below. Under the hood, execution is orchestrated by a [Snakemake](https://snakemake.readthedocs.io) workflow ([workflow/](workflow/)) in which every pipeline tool is its own rule — interrupted scans resume, and individual steps can be re-run without redoing the whole pipeline using [rerunStep.py](source/Run3ModelGen/scripts/rerunStep.py) (original config/seed are recovered from the scan's `scan_config.yaml`):

```bash
rerunStep.py --step micromegas --scan_dir scan     # re-runs micromegas + everything after it
rerunStep.py --step ntuple --scan_dir scan         # only rebuild the ntuple
```

Every run saves its DAG into the scan dir (`dag.dot`, plus `dag.svg` once the [`dagviz` environment](#environments) is installed — `pixi install -a`). Power users can drive the workflow directly with snakemake (`pixi run smk ...`); see [doc/snakemake.md](doc/snakemake.md) for the full guide: config keys, profiles, reproducibility.

## Commands

Everything below is on your `PATH` after `source build/setup.sh`; each takes `--help`.

| Command | What it does |
| --- | --- |
| [`genModels.py`](source/Run3ModelGen/scripts/genModels.py) | Generate models — one scan, or `--num_jobs N` batches, on `--backend local\|slurm\|htcondor`, optionally `--publish`ing to EOS. See [Batch running](#batch-running-slurm--htcondor). |
| [`rerunStep.py`](source/Run3ModelGen/scripts/rerunStep.py) | Re-run one pipeline step of an existing scan plus everything after it, then rebuild the ntuple. |
| [`extractModels.py`](source/Run3ModelGen/scripts/extractModels.py) | Extract selected models for the pMSSMFactory — from one scan dir/tarball, or a whole published run with `--eos_dir`. See [Extracting interesting models](#extracting-interesting-models). |
| [`genConfig.py`](source/Run3ModelGen/scripts/genConfig.py) | Write an example config for a given prior to start from. See [Changing the scan configuration](#changing-the-scan-configuration). |
| [`emcee_diagnostics.py`](source/Run3ModelGen/scripts/emcee_diagnostics.py) | emcee/MCMC convergence diagnostics from `chain.h5` (R̂, bulk/tail ESS, τ, corner/trace/log-prob plots, input-vs-target scatter, burn-in check). Run via `pixi run emcee_diagnostics`. See [Diagnostics](#diagnostics). |

Plus these pixi tasks, which need no sourced environment:

| Task | What it does |
| --- | --- |
| `pixi install -a` | Install all pixi environments (do this once per checkout, before building). |
| `pixi run build` | Configure and compile (micrOMEGAs, pythia8, `build/setup.sh`). |
| `pixi run test` | Offline check suite: unit tests, the sampled-point golden file, and a DAG dry-run of every workflow mode (seconds; identical to what CI runs). |
| `pixi run smk <args>` | Direct snakemake access for cases the scripts do not expose. |
| `pixi run rediagnose <scan_dir>` | Re-run only the emcee diagnostics of an existing `prior: mcmc` scan, in place into `<scan_dir>/diag` (seconds, no sampling). See [doc/emcee.md](doc/emcee.md#re-running-only-the-diagnostics). |
| `pixi run make-refs` / `upload-refs` | Maintainers: regenerate and upload the CI reference files. See [Reference files & CI regression check](#reference-files--ci-regression-check). |

## Sampling priors

Run3ModelGen supports five sampling priors:

| Prior | Description |
| --- | --- |
| `flat` | Draws parameters uniformly at random within specified ranges. |
| `fixed` | Runs the pipeline on a specific set of parameter points. |
| `mcmc` | emcee affine-invariant ensemble sampler guided by physics constraints. See [MCMC sampling](#mcmc-sampling). |
| `grid` | Cartesian product over specified parameter values. See [Grid sampling](#grid-sampling). |
| `importance` | Perturbs known viable seed models with narrow Gaussians. See [Importance sampling](#importance-sampling). |
| `active_learning` | Surrogate-guided sampling: a transformer, DNN or Gaussian process trained on the labelled models picks the next batch. GPU only. See [Active learning](#active-learning). |

The prior is set in the config file via `prior: flat`, `prior: fixed`, `prior: mcmc`, `prior: grid`, `prior: importance`, or `prior: active_learning`.

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
genConfig.py active_learning           # Generates example_active_learning_config.yaml in current dir
```

Example configs are stored in [source/Run3ModelGen/data/configs/](source/Run3ModelGen/data/configs/).

## MCMC sampling

MCMC sampling uses the [emcee](https://emcee.readthedocs.io/en/stable/) affine-invariant ensemble sampler to explore parameter space guided by physics observables. Instead of sampling uniformly, an ensemble of walkers preferentially visits regions where computed observables (e.g. Higgs mass, relic density) agree with experimental measurements. The log-probability is `-chi2/2 + flat_prior`, with chi2 built from the `constraints` block. emcee's stretch move is affine-invariant — it adapts automatically to the local scale and correlations of the target, so there is **no proposal-width tuning**.

To use MCMC, set `prior: mcmc` and define constraints in your config file:

```yaml
prior: mcmc
num_models: 1            # emcee uses one model per worker; keeps the >10000 guard happy

mcmc:
  nwalkers: 64           # ensemble size (>= 2*ndim, even, = 2*nworkers)
  nworkers: 32           # worker-pool size -> CPUs requested for the chain rule
  max_steps: 20000       # iteration cap (convergence usually stops earlier)
  burn_in: 200           # absolute floor (iterations) on the ntuple burn-in
  check_every: 100       # autocorrelation / progress cadence (iterations)
  init:
    strategy: spread     # uniform across the prior (or `ball` around init.center_file)
  moves:
    - {type: StretchMove, kwargs: {a: 2.0}, weight: 1.0}
  record_observables:    # emcee blobs -> ntuple branches
    - {name: m_h, branch: SP_m_h, source: {step: SPheno, file_type: slha, block: MASS, entry: 25}}

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
- **`nwalkers`**: Ensemble size. Use `2 × nworkers`, keep it even, and at least `2 × ndim`.
- **`nworkers`**: Worker-pool size; wired to the `chain` rule's CPUs-per-task.
- **`max_steps`**: Iteration cap (the wall-clock guard / convergence check usually stops earlier).
- **`burn_in`** / **`burn_in_factor`**: Ntuple burn-in = `max(burn_in, ceil(burn_in_factor × max_tau))`.
- **`check_every`**, **`target_autocorr_factor`**, **`tau_rel_tol`**: Autocorrelation cadence and convergence targets (`N > factor × tau`, τ stable to the tolerance).
- **`walltime_hours`**: Wall-clock guard budget (keep below the `chain` rule's runtime).
- **`init.strategy`**: `spread` (uniform across the prior) or `ball` (around `init.center_file`).
- **`moves`**: emcee moves (default `StretchMove(a=2.0)`; a DE-move mix helps harder targets).
- **`record_observables`**: `{name, branch, source}` list recorded as blobs → ntuple branches.

A top-level `tool_timeout` (seconds) wraps each external physics-tool call so a single hung binary can't stall an ensemble half-step. The chain runs the whole pipeline inside sampling and checkpoints to `chain.h5` (an [`emcee.backends.HDFBackend`](https://emcee.readthedocs.io/en/stable/user/backends/)) in the scan dir — this is undeclared side-state, so re-running the `chain` rule **resumes** an unconverged chain instead of wiping it.

### Diagnostics

Convergence is diagnosed from the `chain.h5` backend(s) with `emcee_diagnostics.py`, which reports mean acceptance fraction, integrated autocorrelation time τ per parameter (and the τ-vs-N curve), Gelman-Rubin R̂ and bulk/tail ESS (via arviz, within and across independent ensembles), a burn-in check, and corner/trace/log-prob plots plus an input-vs-target scatter. On all samples pooled it also runs a **rank-statistic uniformity check** (Betancourt's computational faithfulness — ranks pooled across chains must be uniform, and the shape of any departure says *how* the chains disagree; not simulation-based calibration, see [doc/emcee.md](doc/emcee.md#rank-statistic-computational-faithfulness)). It finishes with a **criteria table** — each convergence requirement, why it exists (with an arXiv reference), and whether this scan meets it, opening with a scannable pass/fail SUMMARY — and a CONVERGED / NOT CONVERGED verdict gated on the three hard criteria (`N > 50τ`, R̂ < 1.05, ESS_bulk ≥ 400); see [doc/emcee.md](doc/emcee.md#convergence-checklist). It uses the same τ-scaled discard rule as the sampler's ntuple, and can emit a ROOT `"susy"` ntuple per ensemble so `extractModels` keeps working:

```bash
pixi run emcee_diagnostics \
    --glob "/ptmp/$USER/run3modelgen/<runid>/scan.*/chain.h5" \
    --outdir results/<runid>/diag --to-ntuple results/<runid>/ntuples
```

Aim for a mean acceptance fraction of ~0.2–0.5, `N > 50 × τ` for every parameter, cross-ensemble R̂ < 1.01 (≤ 1.05 acceptable) and ESS_bulk ≫ 400. If `N < 50τ`, re-run the chain (`--forcerun chain`) to add iterations.

For a tuned starting config on relic-density-only scans, see [`example_mcmc_relic_tuned.yaml`](source/Run3ModelGen/data/configs/example_mcmc_relic_tuned.yaml).

For full details, see [doc/emcee.md](doc/emcee.md). For constraint specification, see [doc/constraints.md](doc/constraints.md).

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

## Active learning

`prior: active_learning` labels a uniform seed set with the pipeline, trains a surrogate
model on every labelled point, scores a fresh pool of candidates, selects a batch by an
acquisition rule, labels it, and repeats. Four surrogates (`transformer`, `dnn`,
`exact_gp`, `deep_gp`), two heads (`classification` of which side of the target value a
point falls on, or `regression` of the transformed target) and three acquisition rules
(`cls_entropy`; `entropy_batch`, `tol_only_random`) are available; the default is the
transformer with classification and entropy. It is the loop of the pMSSM active-learning
benchmark ([jwuerzinger/pmssm-al-benchmark](https://github.com/jwuerzinger/pmssm-al-benchmark)),
ported into Run3ModelGen with no dependency on that repository.

```yaml
prior: active_learning
num_models: 500                # selected and labelled per iteration
active_learning:
  environment: rocm            # the GPU pixi environment the loop runs in (rocm | cuda)
  seed_models: 2000
  n_iterations: 40
  model: transformer           # transformer | dnn | exact_gp | deep_gp
  head: classification         # classification | regression
  strategy: cls_entropy        # classification: cls_entropy; regression: entropy_batch | tol_only_random
  target: {branch: MO_Omega, value: 0.12, transform: log, valid: {lo: 0, hi: 1}, hit_tolerance: 0.10}
  vetoes: [{branch: SP_m_h, lo: 0}, {branch: SP_LSP_type, lo: 0.5, hi: 3.5}]
```

**It runs on a GPU, in one of the two extra pixi environments** (`rocm` for AMD, `cuda`
for NVIDIA; both are supersets of the default one), and refuses the default (CPU)
environment with a `FATAL` message — at config validation, at `genModels.py`'s
pre-flight when launched from another environment, and in the job if no GPU is visible:

```bash
pixi install -e rocm && pixi run -e rocm build      # once
pixi run -e rocm genModels.py --config_file source/Run3ModelGen/data/configs/example_active_learning_config.yaml \
    --scan_dir /ptmp/$USER/al_relic --backend slurm
```

Every iteration is an ordinary scan directory (`<scan_dir>/iteration_NNN/` with the
per-tool outputs, `scan_config.yaml` and `ntuple.0.0.root`, plus the training log, the
surrogate checkpoint and the selection), and the scan's ntuple merges them all with
`al_*` branches (iteration, prediction, acquisition score, valid, hit). The loop
checkpoints after every iteration and resumes when the same command is re-run. Full
reference: [doc/active_learning.md](doc/active_learning.md).

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

## Batch running (SLURM / HTCondor)

For larger scans, `genModels.py` fans out over batch jobs — same script, any backend:

```bash
genModels.py --config_file myconfig.yaml --num_jobs 42 --backend htcondor --publish
genModels.py --config_file myconfig.yaml --num_jobs 42 --backend slurm --scan_dir /ptmp/$USER/myrun
```

- Each of the `num_jobs` batches generates `num_models` models (from the config) with its own reproducibly derived seed, as one cluster job, into `<scan_dir>/scan.<runid>.<b>/`. The run is identified by a digits-only `runid` (printed at submission; defaults to a timestamp; `--runid` to pin it) — `(runid, --base_seed, config)` reproducibly determine every sampled point of the run.
- `--publish` (any backend) additionally tars every batch and **moves** tarball + ntuple + config to `$EOSPATH/<runid>/<b>/` — the layout the pMSSMFactory flow consumes. The batch's local scan dir and tarball are deleted once the published copies are verified (present and byte-for-byte the right size), so a published run occupies no space outside EOS: a 300-batch run of 10,000 models would otherwise leave ~340 GB behind, well over an AFS work quota. Only the small `r3mg/` bookkeeping stays, so re-running the same command is still a no-op rather than a regeneration. Pass `--keep_local` to keep the scan dirs too. Without `--publish`, the ntuples under `<scan_dir>` are the product and nothing is deleted.
- **On HTCondor, batch runs write each batch to its own worker's `$_CONDOR_SCRATCH_DIR` instead of `--scan_dir`**, so N concurrent batches need no room on the submit volume and only the published copies leave the node. That makes `--publish` required (scratch dies with the job, so an unpublished run produces nothing) and `--keep_local` refused; both are rejected up front. Pass `--scan_dir` explicitly to opt back into the shared filesystem — `genModels.py` then warns with the projected peak, which is what a 300-batch run needs *simultaneously*, since batches are all submitted at once. See [Batch data on the worker](doc/snakemake.md#batch-data-on-the-worker).
- On HTCondor the job flavour is set with `--flavour` (e.g. `espresso` for testing); SLURM partition/runtime defaults live in [workflow/profiles/slurm/config.yaml](workflow/profiles/slurm/config.yaml).
- Cluster submissions detach a Snakemake orchestrator on the submit host (log: printed at submission; `--foreground` to stay attached). If it dies, running jobs finish on their own — **re-run the exact same command to resume**; finished batches are skipped. Changing the config or `--base_seed` under an existing `--runid` is refused (it would re-sample); use a new runid instead.

### Extracting models from a published run

A published run ($EOSPATH/<runid>/ with per-batch subdirs) is extracted batch-by-batch with the same script that handles single scans:

```bash
extractModels.py --eos_dir /eos/user/j/jwuerzin/Run3ModelGen/15280231 --backend htcondor --selection "akarr['SP_m_h'] < 125."
```

One extraction job per ProcId subdirectory; the output `Models` directory can be fed into the pMSSMFactory directly and is stored in `$EOSPATH/SelectedModels/<ClusterId>/`. Backend and flavour flags work exactly as for `genModels.py`.

# Documentation

Detailed documentation is available in the [doc/](doc/) directory:

- [doc/snakemake.md](doc/snakemake.md) - Snakemake workflow: entrypoint flag reference (`--backend`, batch mode, resume), rule graph, profiles (local/slurm/htcondor), re-running steps, reproducibility, DAG snapshots
- [workflow/README.md](workflow/README.md) - Workflow *sources* layout (for contributors): which rule lives in which `.smk`, the `r3mg_batch.py` subcommand interface, and the invariants a change must respect
- [doc/ntuple.md](doc/ntuple.md) - NTuple structure: tree layout, branch naming, and per-tool branch groups
- [doc/emcee.md](doc/emcee.md) - MCMC (emcee) sampling: algorithm, configuration keys, run commands, diagnostics, resume, and output
- [doc/active_learning.md](doc/active_learning.md) - Active learning: the loop, the GPU environments and the FATAL rule, every config key, surrogates/heads/strategies, the per-iteration output layout, resume
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

## Unlocking an interrupted scan

Snakemake locks its working directory (the scan parent — `run/` for the local backend)
while a run is in progress. If a run is killed — Ctrl-C, a crash, a node/power loss, or
two scans launched into the same directory at once — a stale lock can be left behind, and
the next run fails with `LockException: ... Directory cannot be locked`.

With no Snakemake actually running, clear the lock and re-run — either delete it directly:
```bash
rm -rf run/.snakemake/locks
```
or use Snakemake's `--unlock` (paths as the local backend launches it, from `run/`):
```bash
pixi run smk --workflow-profile ../workflow/profiles/local --unlock \
    --config mode=single scan_dir=<scan dir> config_file=<config>
```
Avoid launching two scans into the same scan parent (`run/`) at once — the most common
way to trip the lock.

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
  - `EOSPATH`: Points to your EOS output directory (a pre-set `$EOSPATH` wins over the default — e.g. to `--publish` to a non-EOS path)
  - `RUN3MODELGENDIR`: Points to the repository root directory
  - `CONDOR_CONFIG`: set to `/dev/null` **only** on hosts with no HTCondor client, to silence the `htcondor2` config warning that snakemake's plugin discovery would otherwise print on every start. On lxplus (where `condor_submit` exists) it is left untouched so normal config discovery applies.

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
