# pixi Package Manager

This document describes how Run3ModelGen uses [pixi](https://pixi.sh/) to manage dependencies, environments, and tasks. pixi is a fast, cross-platform package manager built on the conda ecosystem.

## Quick reference

| Action | Command |
| --- | --- |
| Install pixi | `curl -fsSL https://pixi.sh/install.sh \| bash` |
| Enter the default shell | `pixi shell` |
| Build the project | `pixi run build` |
| Generate models | `pixi run genModels` |
| Clean build & run dirs | `pixi run clean` |
| Enter the ROOT environment | `pixi shell -e rootenv` |
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

Run3ModelGen defines two pixi environments:

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

For more on pixi multi-environment support, see the [pixi environments documentation](https://pixi.sh/latest/features/multi-environment/).

## Tasks

pixi tasks provide reproducible, single-command workflows. They are defined in `pixi.toml` under `[tasks]`.

### Available tasks

| Task | Command | Description |
| --- | --- | --- |
| `build` | `pixi run build` | Configures and compiles the project with cmake. On Linux, uses all available cores minus one. |
| `genModels` | `pixi run genModels` | Sources the build environment, creates a `run/` directory, and runs `genModels.py`. |
| `extractModels` | `pixi run extractModels` | Sources the build environment and runs `extractModels.py` with default arguments. |
| `clean` | `pixi run clean` | Removes the `build/` and `run/` directories. |
| `genModels-clean` | `pixi run genModels-clean` | Runs `clean`, `build`, and `genModels` in sequence (full rebuild + generate). |
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
| Compilers | `gcc`, `gfortran`, `cxx-compiler` (Linux only) |
| Build tools | `cmake`, `make`, `automake`, `patch` |
| C++ libraries | `boost`, `eigen`, `libconfig` (Linux only) |
| Python | `python 3.11` |
| Python packages | `pyyaml`, `pyslha`, `uproot`, `pandas`, `scipy`, `click`, `rich`, `structlog`, `colorama`, `ipykernel` |
| Physics | `spheno-atlas`, `softsusy`, `superiso`, `feynhiggs`, `gm2calc-python`, `smodels` |
| Utilities | `which`, `git` |

**rootenv environment:**

| Package | Version |
| --- | --- |
| `python` | 3.11.* |
| `root_base` | >= 6.40.0 |

## Channels

Run3ModelGen uses the [conda-forge](https://conda-forge.org/) channel exclusively:

```toml
channels = ["conda-forge"]
```

conda-forge is a community-maintained collection of conda packages. All dependencies, including compilers and physics software, are sourced from this channel.

## Platforms

Currently only `linux-64` is supported:

```toml
platforms = ["linux-64"]
```

Support for `osx-64` and `osx-arm64` is planned but not yet available (the external physics software needs porting).

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
