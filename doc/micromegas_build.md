# Adding micromegas to the framework

This document describes how micromegas is integrated into the Run3ModelGen build system.

## Integration Approach

Micromegas is built using CMake's `ExternalProject_Add`, which:
1. Downloads the official micromegas 5.2.1 tarball from LAPTH
2. Applies patches to support building with pixi environment
3. Builds the package out-of-source

## Patches Applied

Three patches are applied to the official source:

1. **001-calchep-compiler-flags.patch.in**: Adds pixi compiler flags for gfortran linking
   - Adds `-fcommon` flag (required for GCC 10+ to handle multiple definitions)
   - Adds library path to pixi environment: `-L{PIXI_LIB_PATH}`
   - Links Fortran libraries: `-lgfortran -lquadmath`
   - Uses CMake `configure_file()` to substitute absolute path at build time

2. **002-mssm-disable-features.patch**: Disables unnecessary features
   - Comments out: RGE suspect, EWSB, MASSES_INFO, LILITH, SMODELS, DECAYS
   - Streamlines build for essential functionality

3. **003-mssm-tabular-output.patch**: Formats output as tab-separated
   - Modifies printf statements for easier parsing
   - Changes output from human-readable to machine-readable format

## Patch Location

Patches are stored in `source/patches/micromegas/`

## Build Process

The ExternalProject_Add performs the following steps:
1. **Download**: Fetches micromegas_5.2.1.tgz from LAPTH
2. **Patch**: Applies the three patches in order
3. **Configure**: Creates empty work directories required by micromegas
4. **Build**: Runs `make` for base micromegas, then `make` for MSSM module

## Rebuilding from Scratch

To force a clean rebuild of micromegas:
```bash
rm -rf build/micromegas_5.2.1
pixi run build
```

## Modifying Patches

To update patches:

1. Download original tarball:
   ```bash
   wget https://lapth.cnrs.fr/micromegas/downloadarea/code/micromegas_5.2.1.tgz
   tar -xzf micromegas_5.2.1.tgz
   ```

2. Make changes to extracted files

3. Generate patch:
   ```bash
   cd micromegas_5.2.1
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
└── micromegas_5.2.1/
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
