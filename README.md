# Run3ModelGen <!-- omit in toc -->

Draft repo for updated shiny pMSSM model generation. Aiming for this repo to be fully independent of HTCondor & lxplus as well as being fully Dockerised.

# Table of Contents <!-- omit in toc -->

- [Targets](#targets)
  - [Software content](#software-content)
  - [Functionalities](#functionalities)
- [Installation](#installation)
  - [pixi](#pixi)
  - [Run3ModelGen](#run3modelgen)
- [On startup](#on-startup)
- [Running](#running)
  - [Changing the scan configuration](#changing-the-scan-configuration)
  - [Extracting interesting models](#extracting-interesting-models)
    - [From scan directories](#from-scan-directories)
    - [From .tar balls directly](#from-tar-balls-directly)
  - [HTCondor](#htcondor)
    - [Generating models](#generating-models)
    - [Extracting models](#extracting-models)
- [FAQ](#faq)
  - [Adding software/updating the software version](#adding-softwareupdating-the-software-version)
  - [Fixing the parameter points](#fixing-the-parameter-points)
  - [Gaps in the scan ranges](#gaps-in-the-scan-ranges)
  - [cmake_build_env.sh](#cmake_build_envsh)
  - [pixi](#pixi-1)
  

# Targets
- [x] Move from shell scripts to Makefile structure for building
- [x] migrate to alma9
- [x] Dockerise
- [x] Add CI
- [ ] Update software content
- [ ] Add missing functionalities
- [ ] Make compatible with GAMBIT scanners
- [ ] Improve overall readability & Useability
- [x] Add release notes incl. versions for external software.
- [x] Make venv to not clutter the user's local AB python -> solved with pixi!

## Software content

| Software | Implemented Version | Latest Version | 
| - | - | - | 
| SPheno | 4.0.6 | 4.0.6 | 
| Softsusy | 4.1.22 | 4.1.22 | 
| FeynHiggs | 2.19.0 | 2.19.0 | 
| SuperIso | 4.0 | 5.0 | 
| GM2Calc | 2.0.0 | 2.2.0 |
| micrOMEGAs | 5.2.1 | 6.0 | 
| EVADE | - | 0.2.0 | 

EVADE broke after a recent update to HOM4PS and will remain unavailable for the time being.

## Functionalities

- [x] Script to run flat scan & save output models
- [x] Docker images of framework
- [x] Ntuple making
  - [x] Add ability to specify steps and their outputs diverging from defaults.
- [x] Model selection for factory
- [x] HTCondor support for larger parallel scans (using Docker?!)
- [ ] Support for non-alma9 machines (e.g. MacOS)
- [ ] Add support for re-running individual steps
- [ ] Add support for generating additional models to existing scan
- [x] Improve movemodels .tar ball handling

# Installation

## pixi

This repo uses [pixi](https://pixi.sh/latest/) to manage packages, compilers, etc. Hopefully this will only make things easier for you as a user, but it does require installing pixi locally before we can compile the Run3ModelGen code. To install pixi, do:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bash_profile 
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

# On startup

As perviously mentioned, pixi should be ready to use as per your `.bash_profile`, so there is no need to do anything here. 
For settup up the Run3ModelGen, please simply execute:

```bash
pixi shell
source build/setup.sh
```
And you're good to go!

# Running

The main script for generating models is [genModels.py](source/Run3ModelGen/scripts/genModels.py). If you set everything up correctly, this can simply be executed with:

```bash
genModels.py
```

The script will save the created model files and the corresponding root ntuple.

## Changing the scan configuration

If you would like to change the scan configuration or seed, simply run

```bash
genModels.py --config_file myconfig.yaml --seed 42 
```

where `myconfig.yaml` is your config file containing the scan configuration. If no config file is supplied, the [default config file](source/Run3ModelGen/data/default_config.yaml) will be read. All values that are missing from `myconfig.yaml` are read from the default as well.

Here the contents of this default configuration:

```yaml
scan_dir: scan
prior: flat
num_models: 3
isGMSB: False
parameters:
  tanb: [1, 60]
  M_1: [-2000, 2000]
  M_2: [-2000, 2000]
  M_3: [4000, 4000]
  AT: [-8000, 8000]
  Ab: [-2000, 2000]
  Atau: [-2000, 2000]
  mu: [-2000, 2000]
  mA: [2000, 2000]
  meL: [0, 2000]
  mtauL: [2000, 2000]
  meR: [0, 2000]
  mtauR: [2000, 2000]
  mqL1: [4000, 4000]
  mqL3: [4000, 4000]
  muR: [4000, 4000]
  mtR: [4000, 4000]
  mdR: [4000, 4000]
  mbR: [4000, 4000]
steps:
  - name: prep_input
    output_dir: input
    prefix: IN
  - name: SPheno
    input_dir: input
    output_dir: SPheno
    log_dir: SPheno_log
    prefix: SP
  - name: softsusy
    input_dir: input
    output_dir: softsusy
    prefix: SS
  - name: micromegas
    input_dir: SPheno
    output_dir: micromegas
    prefix: MO
  - name: superiso
    input_dir: SPheno
    output_dir: superiso
    prefix: SI
  - name: gm2calc
    input_dir: SPheno
    output_dir: gm2calc
    prefix: GM2
  - name: evade
    input_dir: SPheno
    output_dir: evade
    prefix: EV
```

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

# FAQ

More will follow soon. In the meantime, if you have any questions at all, please contact jonas.wuerzinger@cern.ch.

## Adding software/updating the software version

Unfortunately, most software included here uses in-source-builds exclusively. I have taken notes on how I added the different software in this cmake workflow in:
- [doc/spheno_build](doc/spheno_build.md)
- [doc/softsusy_build](doc/softsusy_build.md)
- [doc/micromegas_build](doc/micromegas_build.md)
- [doc/superiso_build](doc/superiso_build.md)

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
  - `PATH`: Adds all external software binaries (SPheno, softsusy, micromegas, etc.) and Run3ModelGen scripts
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

If you wish to edit the pixi package content, you can either edit the `pixi.toml` file directly, or use (some of) these handy commands which I used to initiate the repo:

```bash
# Install pixi & initialise
curl -fsSL https://pixi.sh/install.sh | bash
. ~/.bash_profile 
pixi init .

# general dependencies
time pixi add 'python=3.11'
time pixi add gfortran
time pixi add cc
time pixi add gcc
time pixi add cxx-compiler
time pixi add cmake
time pixi add make
time pixi add which

# python packages on conda
time pixi add numpy
time pixi add pyyaml
time pixi add pyslha

# start shell
pixi shell
```

Please refrain from editing the `pixi.lock` file directly!