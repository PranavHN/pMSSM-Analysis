# Adding micromegas to the framework

This document describes how micromegas is integrated into the Run3ModelGen build system.

## Integration Approach

Micromegas is built using CMake's `ExternalProject_Add`, which:
1. Downloads the official micromegas 7.1.4 tarball from micromegasdm.github.io
2. Applies patches to support building with pixi environment
3. Builds the package out-of-source

## Patches Applied

Three patches are applied to the official source:

1. **001-calchep-compiler-flags.patch.in**: Adds pixi compiler flags for gfortran linking
   - Adds `-fcommon` flag (required for GCC 10+ to handle multiple definitions)
   - Adds library path to pixi environment: `-L{PIXI_LIB_PATH}` (+ `-Wl,-rpath` so the
     built binaries find gfortran/quadmath at run time, needed on macOS)
   - Links Fortran libraries: `-lgfortran -lquadmath`
   - Adds `-fsigned-char` and clang-only `-Wno-error=...` flags (gated behind a
     `$CC --version | grep clang` check) so the legacy CalcHEP C compiles with modern
     clang on macOS without affecting the Linux gcc build
   - Uses CMake `configure_file()` to substitute absolute path at build time

2. **002-mssm-disable-features.patch**: Disables unnecessary features
   - Comments out: RGE suspect, EWSB, MASSES_INFO, CDM_NUCLEUS, DECAYS
     (LILITH and SMODELS are already disabled by default in micrOMEGAs 7.1.4)
   - CDM_NUCLEUS is disabled because 7.1.4's `MSSM/main.c` references a removed direct-
     detection experiment identifier (`LZ5T_2024`) there and fails to compile; its output
     is not used by the framework (the CDM_NUCLEON proton/neutron cross sections are kept)
   - Streamlines build for essential functionality

3. **003-mssm-tabular-output.patch**: Formats output as tab-separated
   - Modifies printf statements for easier parsing
   - Changes output from human-readable to machine-readable format

## Patch Location

Patches are stored in `source/patches/micromegas/`

## Build Process

The ExternalProject_Add performs the following steps:
1. **Download**: Fetches micromegas_7.1.4.tgz from micromegasdm.github.io
2. **Patch**: Applies the three patches in order
3. **Configure**: Creates empty work directories required by micromegas
4. **Build**: Runs `make` for base micromegas, then `make` for MSSM module
5. **Install (post-build)**: On macOS, rewrites the `maxGap.so` install names to absolute
   paths (`fix_micromegas_install_names.sh`); then warms the CalcHEP process-library cache
   (`warm_micromegas_cache.sh`).

## Process-library cache warming

The first time micrOMEGAs' `main` evaluates a model, CalcHEP generates and compiles the
shared libraries for every needed matrix element (relic-density annihilation/coannihilation
channels and the direct-detection nucleon amplitudes) into `MSSM/work/so_generated/`. This
takes several minutes. Those libraries are **process-level, not parameter-point-level**, so
they are reused by every subsequent model and every subsequent `genModels` run against the
same build.

To avoid paying this on the first scan point of every run (in particular on a fresh CI
runner, where it would otherwise hit each job), the build runs `main` once on the
micrOMEGAs-shipped sample spectrum (`MSSM/input.slha`) via `warm_micromegas_cache.sh`. The
resulting libraries are cached in the build tree — and, because the Docker image is built by
running `pixi run build`, baked into the CI image. The step is **best-effort** (a failure
does not fail the build; the libraries would simply be built at run time instead) and
**idempotent** (skipped if the libraries are already present). Its CalcHEP compilation
output is streamed to the build log so you can see what is being compiled.

Note: warming uses one representative spectrum, so scan points that need coannihilation
channels absent from that spectrum will still compile those few extra processes on first use.

## Rebuilding from Scratch

To force a clean rebuild of micromegas:
```bash
rm -rf build/micromegas_7.1.4
pixi run build
```

## Modifying Patches

To update patches:

1. Download original tarball:
   ```bash
   wget https://micromegasdm.github.io/downloadarea/v7.1/micromegas_7.1.4.tgz
   tar -xzf micromegas_7.1.4.tgz
   ```

2. Make changes to extracted files

3. Generate patch:
   ```bash
   cd micromegas_7.1.4
   diff -u path/to/original_file path/to/modified_file > ../source/patches/micromegas/patch_name.patch
   ```

4. For the compiler flags patch, use `@PIXI_LIB_PATH@` placeholder:
   ```bash
   # Replace the hardcoded path with placeholder
   sed -i 's|/path/to/.pixi/envs/default/lib|@PIXI_LIB_PATH@|g' modified_file
   # Then create patch with .in suffix
   diff -u original_file modified_file > ../source/patches/micromegas/001-calchep-compiler-flags.patch.in
   ```

5. Test patch application:
   ```bash
   patch -p1 --dry-run < ../source/patches/micromegas/patch_name.patch
   ```

## Directory Structure

After build, micromegas is located at:
```
build/
└── micromegas_7.1.4/
    ├── src/
    │   └── micromegas/          # Extracted and patched source
    │       ├── MSSM/
    │       │   └── main         # Executable
    │       └── ...
    └── stamp/                   # Build stamps for incremental builds
```

The MSSM executable is added to PATH via `setup.sh`.

## Historical Notes

### Original Approach (Deprecated)

Previously, micromegas was stored as a patched version in the git repository at `source/micromegas_5.2.1/`, which:
- Tracked 1,400+ files in git (~13MB)
- Used `add_subdirectory()` for integration
- Made it difficult to see what modifications were made
- Required tracking binary build artifacts

### Migration Benefits

The new ExternalProject approach provides:
- Cleaner git history (patches vs. full source)
- Uses official source distribution
- Clear separation between original code and local modifications
- Easier maintenance and updates
- Smaller repository size
