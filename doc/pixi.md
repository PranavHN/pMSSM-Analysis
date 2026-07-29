# pixi Package Manager

This document describes how Run3ModelGen uses [pixi](https://pixi.sh/) to manage dependencies, environments, and tasks. pixi is a fast, cross-platform package manager built on the conda ecosystem.

## Quick reference

| Action | Command |
| --- | --- |
| Install pixi | `curl -fsSL https://pixi.sh/install.sh \| bash` |
| Enter the default shell | `pixi shell` |
| Build the project | `pixi run build` |
| Generate models | `pixi run genModels` |
| Drive snakemake directly | `pixi run smk <snakemake args>` |
| Clean build & run dirs | `pixi run clean` |
| Enter the ROOT environment | `pixi shell -e rootenv` |
| Install the DAG-rendering env | `pixi install -e dagviz` (also done by `pixi run build`) |
| Add a dependency | `pixi add <package>` |
| List installed packages | `pixi list` |

## Installation

pixi is installed via a single shell command:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bash_profile
```

This adds pixi to your `PATH` via `~/.bash_profile`. On subsequent logins, pixi is available automatically.

For alternative installation methods (Homebrew, Windows, etc.), see the [pixi installation guide](https://pixi.sh/latest/getting-started/).

## Key files

| File | Description | Edit? |
| --- | --- | --- |
| `pixi.toml` | Project configuration: dependencies, tasks, environments, platforms. | Yes |
| `pixi.lock` | Exact resolved dependency versions. Generated automatically. | **Never edit manually.** |
| `.pixi/` | Local environment directory (created by `pixi install`). | No (gitignored) |

## Environments

Run3ModelGen defines three pixi environments:

### Default environment

The default environment contains everything needed to build and run the full physics pipeline: Python 3.11, Fortran and C++ compilers, cmake, all Python dependencies (pyslha, uproot, scipy, click, structlog, etc.), and the external physics libraries.

```bash
pixi shell          # Enter the default environment
pixi run build      # Build without entering a shell
```

This is the environment you use for day-to-day development and model generation.

### `rootenv` environment

The `rootenv` environment is a lightweight, standalone environment containing only Python 3.11 and ROOT (>= 6.40, via `root_base`). It is defined with `no-default-feature = true`, meaning it does **not** inherit any packages from the default environment (no compilers, no physics tools).

```bash
pixi shell -e rootenv     # Enter the ROOT environment
pixi run -e rootenv python my_script.py  # Run a script in rootenv
```

**When to use `rootenv`:**
- Analysing ROOT ntuples produced by the scan
- Running ROOT macros or PyROOT scripts
- Any task that needs ROOT but not the full build toolchain

This separation keeps the ROOT environment fast to resolve and avoids potential dependency conflicts between ROOT and the build toolchain.

**Configuration in `pixi.toml`:**
```toml
[feature.rootenv.dependencies]
python = "3.11.*"
root_base = ">=6.40.0,<7"

[environments]
rootenv = { features = ["rootenv"], no-default-feature = true }
```

### `dagviz` environment

A standalone graphviz install used to render the workflow's DAG snapshots
(`snakemake --dag | pixi run -e dagviz dot -Tsvg > dag.svg`; the automatic
per-run `dag.svg` also uses it when installed — see
[snakemake.md](snakemake.md)). It is deliberately isolated: graphviz's GUI
dependency stack cannot coexist with xrootd 6.x. Neither package depends on the
other; they compete over `libxml2`, of which an environment holds exactly one:

    graphviz 12.2.1  -> gtk3, pango, which are built against libxml2 <2.14
    xrootd   6.1.0   -> libxml2-16 >=2.14.6, libcurl >=8.21.0, krb5 >=1.22.2
    xrootd   5.8.4   -> libxml2 >=2.13.8,<2.14, libcurl >=8.14.1, krb5 >=1.21.3

So adding graphviz to the default environment makes xrootd 6.1.0 unsolvable and
the solver quietly falls back to the 5.x line. Measured with a trial solve:
xrootd 6.1.0 -> 5.8.4, libcurl 8.21 -> 8.18, krb5 1.22.2 -> 1.21.3 (plus
libxml2, libllvm19 and zeromq), and 40 added packages (gtk3, cairo, pango,
gdk-pixbuf, libcups). xrootd is what `--publish` uses to reach EOS, so that
trade is not worth a rendered graph.

`pixi run build` installs this environment as part of a normal build; the
standalone command is `pixi install -e dagviz` (once per machine/checkout).

For more on pixi multi-environment support, see the [pixi environments documentation](https://pixi.sh/latest/features/multi-environment/).

## Tasks

pixi tasks provide reproducible, single-command workflows. They are defined in `pixi.toml` under `[tasks]`.

### Available tasks

| Task | Command | Description |
| --- | --- | --- |
| `build` | `pixi run build` | Configures and compiles the project with cmake. On Linux, uses all available cores minus one. |
| `genModels` | `pixi run genModels` | Sources the build environment, creates a `run/` directory, and runs `genModels.py`. |
| `extractModels` | `pixi run extractModels` | Sources the build environment and runs `extractModels.py` with default arguments. |
| `smk` | `pixi run smk <args>` | Direct snakemake access for power users: sources the build environment, runs from `run/`, passes all arguments through to `snakemake -s ../workflow/Snakefile` (see [snakemake.md](snakemake.md)). |
| `test` | `pixi run test` | The full offline check suite via `tests/run_checks.sh`: unit tests (`tests/`, including the sampled-point golden file) and workflow DAG dry-runs in every mode. Seconds, no physics tools or cluster needed. The CI dry-run job runs the *same script*, so local and CI cannot drift. Requires a prior `pixi run build` (the package is assembled under `build/`). |
| `clean` | `pixi run clean` | Removes the `build/` and `run/` directories. |
| `genModels-clean` | `pixi run genModels-clean` | Runs `clean`, `build`, and `genModels` in sequence (full rebuild + generate). |
| `make-refs` | `pixi run make-refs` | Produces the CI reference scans and prints their per-step success counts (manual, no upload). |
| `upload-refs` | `pixi run upload-refs` | Uploads the references produced by `make-refs` to the shared EOS area (manual, needs credentials — see the README). |
| `setup-pythia8` | `pixi run setup-pythia8` | Repairs/builds SModelS's bundled pythia8 (see [pythia8.md](pythia8.md)). Normally runs automatically during the build and on `source build/setup.sh`. |

### Platform-specific overrides

The `build` task has a Linux-specific override under `[target.linux-64.tasks]` that uses `nproc` to detect the number of available CPU cores and adds `--clean-first` to ensure a fresh build:

```toml
[target.linux-64.tasks]
build = """
cmake -S source -B build && \
cmake \
    --build build \
    --clean-first \
    --parallel "$(nproc --ignore=1)"
"""
```

On other platforms, the generic `[tasks]` definition is used with a fixed parallelism of 8.

### Running tasks

```bash
pixi run build              # Run the build task
pixi run genModels          # Generate models
pixi run genModels-clean    # Clean + build + generate in one step
```

Tasks can also be run from outside the pixi shell — pixi activates the correct environment automatically.

For more on pixi tasks, see the [pixi tasks documentation](https://pixi.sh/latest/features/advanced_tasks/).

## Dependencies

Dependencies are declared in `pixi.toml` under `[dependencies]` (all platforms) and `[target.linux-64.dependencies]` (Linux-only).

### Adding a dependency

```bash
pixi add <package>                 # Add to default dependencies
pixi add <package> --feature rootenv  # Add to the rootenv feature
```

This updates both `pixi.toml` and `pixi.lock`. Always commit both files.

### Removing a dependency

```bash
pixi remove <package>
```

### Listing installed packages

```bash
pixi list                    # List packages in the default environment
pixi list -e rootenv         # List packages in rootenv
```

### Current dependencies

**Default environment** (build + runtime):

| Category | Packages |
| --- | --- |
| Compilers | `gcc` (Linux only), `gfortran`, `cxx-compiler` |
| Build tools | `cmake`, `make`, `automake`, `patch` |
| C++ libraries | `boost`, `eigen`, `libconfig` |
| Python | `python 3.11` |
| Python packages | `pyyaml`, `pyslha`, `uproot`, `pandas`, `scipy`, `numpy` (pinned — see below), `click`, `rich`, `structlog`, `colorama`, `ipykernel` |
| Physics | `spheno-atlas`, `softsusy`, `superiso`, `feynhiggs` (Linux only), `gm2calc-python`, `smodels` |
| Workflow | `snakemake-minimal`, `snakemake-executor-plugin-slurm` (Linux only), `snakemake-executor-plugin-htcondor` (Linux only, via PyPI — its conda dependency `python-htcondor` has no py3.11 build) |
| Utilities | `which`, `git`, `xrootd` (provides `xrdcp` for the EOS reference files) |

**Why numpy is pinned exactly:** the sampled parameter points come from
`np.random` streams, and the CI regression check compares per-step success
counts against frozen references — a numpy version bump could silently change
every sampled point. Bump the pin only together with a reference regeneration
(`make-refs`/`upload-refs`).

**rootenv environment:**

| Package | Version |
| --- | --- |
| `python` | 3.11.* |
| `root_base` | >= 6.40.0 |

## Channels

```toml
channels = ["conda-forge", "bioconda"]
```

Channel order is priority order: every pre-existing package resolves from
[conda-forge](https://conda-forge.org/) (compilers, physics software, Python
stack); [bioconda](https://bioconda.github.io/) only supplies the snakemake
workflow engine and the SLURM executor plugin. The HTCondor executor plugin
comes from PyPI (`[target.linux-64.pypi-dependencies]`) because its conda
dependency `python-htcondor` has no Python 3.11 build.

## Platforms

```toml
platforms = ["linux-64", "osx-64"]
```

`osx-64` also runs under Rosetta on Apple Silicon; native `osx-arm64` is not
supported (no conda-forge feynhiggs build for macOS, and the pinned
gcc/gfortran 13.2 toolchain only exists for linux-64/osx-64). On macOS the
`feynhiggs` pipeline step is dropped from the workflow automatically, and only
the local execution backend is available (the executor plugins are
linux-only).

## Common workflows

### First-time setup

```bash
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bash_profile
cd Run3ModelGen
pixi shell
cmake -S source -B build
cmake --build build -j8
source build/setup.sh
```

### Daily startup

```bash
cd Run3ModelGen
pixi shell
source build/setup.sh
```

### Updating dependencies

```bash
pixi update                # Update all packages to latest compatible versions
pixi update <package>      # Update a specific package
```

This regenerates `pixi.lock`. Commit both `pixi.toml` and `pixi.lock` after updating.

### Recreating the environment

If the environment becomes corrupted or you want a clean start:

```bash
rm -rf .pixi
pixi install
```

## Troubleshooting

### `pixi: command not found`

Ensure pixi is installed and your shell profile is sourced:
```bash
. ~/.bash_profile
```

### Slow environment resolution

The first `pixi shell` or `pixi install` after cloning may take a few minutes as pixi resolves and downloads all dependencies. Subsequent activations are fast.

### Lock file conflicts

If `pixi.lock` has merge conflicts, the simplest fix is to accept either version and run:
```bash
pixi install
```
This regenerates the lock file from the current `pixi.toml`.

## Further reading

- [pixi documentation](https://pixi.sh/latest/) — full reference
- [pixi.toml reference](https://pixi.sh/latest/reference/pixi_toml/) — all configuration options
- [pixi tasks](https://pixi.sh/latest/features/advanced_tasks/) — task definitions and dependencies
- [pixi environments](https://pixi.sh/latest/features/multi-environment/) — multi-environment support
- [conda-forge](https://conda-forge.org/) — the package channel used by this project
