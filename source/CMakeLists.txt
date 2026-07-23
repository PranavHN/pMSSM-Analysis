# Main CMakeLists.txt for the Run3ModelGen project

# Mandatory setting for minimal CMake requirement
cmake_minimum_required( VERSION 3.20 )

# Create a project
project( Run3ModelGen )

include( ExternalProject )

# Configure micromegas patches with absolute pixi path.
# Use the active conda/pixi env prefix (set by `pixi run`/`pixi shell`) rather than a
# path relative to CMAKE_SOURCE_DIR: the env lives at <repo>/.pixi, but CMAKE_SOURCE_DIR
# is <repo>/source (cmake -S source), so a "${CMAKE_SOURCE_DIR}/.pixi" path is wrong.
# On Linux the conda gcc wrapper puts the env lib on LIBRARY_PATH so the bad -L was
# harmless; on macOS clang does not, so -lgfortran/-lquadmath failed to link.
if(DEFINED ENV{CONDA_PREFIX})
    set(PIXI_LIB_PATH "$ENV{CONDA_PREFIX}/lib")
else()
    set(PIXI_LIB_PATH "${CMAKE_SOURCE_DIR}/../.pixi/envs/default/lib")
endif()
configure_file(
    ${CMAKE_SOURCE_DIR}/patches/micromegas/001-calchep-compiler-flags.patch.in
    ${CMAKE_BINARY_DIR}/patches/001-calchep-compiler-flags.patch
    @ONLY
)

# Add ExternalProjects for tools that run out-of-the-box

# Add micromegas as ExternalProject
ExternalProject_Add( micromegas
    PREFIX micromegas_5.2.1
    URL https://lapth.cnrs.fr/micromegas/downloadarea/code/micromegas_5.2.1.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
    UPDATE_COMMAND ""

    # Apply patches
    PATCH_COMMAND
        patch -p1 < ${CMAKE_BINARY_DIR}/patches/001-calchep-compiler-flags.patch &&
        patch -p1 < ${CMAKE_SOURCE_DIR}/patches/micromegas/002-mssm-disable-features.patch &&
        patch -p1 < ${CMAKE_SOURCE_DIR}/patches/micromegas/003-mssm-tabular-output.patch

    # Create empty directories (needed by micromegas to build)
    CONFIGURE_COMMAND ""

    # Build micromegas base first, then MSSM
    BUILD_COMMAND
        ${CMAKE_MAKE_PROGRAM} -C <SOURCE_DIR> -j1
        COMMAND ${CMAKE_MAKE_PROGRAM} -C <SOURCE_DIR>/MSSM -j1

    BUILD_IN_SOURCE 1

    # macOS: micromegas links MSSM/main against a relative dylib path ("../lib/maxGap.so"),
    # which dyld cannot resolve when `main` runs from the scan directory. Rewrite the
    # install names to absolute paths. The helper is a no-op on non-Darwin platforms.
    INSTALL_COMMAND bash ${CMAKE_SOURCE_DIR}/Run3ModelGen/scripts/fix_micromegas_install_names.sh <SOURCE_DIR>
)

# # Add EVADE as ExternalProject - DISABLED; PROBABLY PERMANENTLY BROKEN!
# ExternalProject_Add( evade
#     PREFIX EVADE-0.2.0
#     GIT_REPOSITORY https://gitlab.com/jonaswittbrodt/EVADE.git
#     DOWNLOAD_EXTRACT_TIMESTAMP NEW
#     CMAKE_ARGS -Wno-dev
#     UPDATE_COMMAND ""
#     BUILD_COMMAND ${CMAKE_COMMAND} -E env "CMAKE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_MAKE_PROGRAM}
#     INSTALL_COMMAND ""
# )

# Repair/build SModelS's bundled pythia8 (works around SModelS's broken pythia8
# auto-download; see source/Run3ModelGen/scripts/setup_pythia8.sh). ALL so it runs
# on a plain `cmake --build build` / `pixi run build`; the script is idempotent.
# Best-effort: pythia8 is only needed at *run* time (the SModelS step), so a failed
# pre-build (e.g. on a memory-limited CI runner) must NOT fail the whole build — it
# warns and SModelS falls back to building pythia8 on first use.
add_custom_target( setup_pythia8 ALL
    COMMENT "Repairing/building SModelS bundled pythia8 (best-effort)"
    COMMAND bash -c "${CMAKE_SOURCE_DIR}/Run3ModelGen/scripts/setup_pythia8.sh || echo '[setup_pythia8] WARNING: build-time pre-build failed; SModelS will build pythia8 at first use'"
)
# Run after micromegas (the only remaining source-built tool; the others now come
# from conda-forge) so it doesn't compete for memory during a parallel `cmake --build`.
add_dependencies( setup_pythia8 micromegas )

# Add shell script to place all relevant files in build
add_custom_target( Run3ModelGen
    COMMENT "Building Run3ModelGen"
    COMMAND ${CMAKE_SOURCE_DIR}/cmake_build_env.sh -s ${CMAKE_SOURCE_DIR} -b ${CMAKE_BINARY_DIR}
)

# make Run3ModelGen the default target for "make"
add_custom_target(default_target DEPENDS Run3ModelGen)
