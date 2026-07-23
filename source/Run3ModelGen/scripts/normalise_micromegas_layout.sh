#!/usr/bin/env bash
#
# Normalise the extracted micrOMEGAs source tree so that the version's files always live
# directly under <source_dir>, regardless of how the archive happened to extract.
#
# The v7.1 tarball was packaged on macOS, so its archive root contains TWO top-level
# entries: the real `micromegas_<version>/` directory and an AppleDouble metadata sidecar
# `._micromegas_<version>`. CMake's ExternalProject only flattens the extracted archive
# into <source_dir> when there is exactly ONE top-level entry, so the platforms diverge:
#   * Linux  — libarchive extracts the `._` sidecar as a real file, so there are two
#              top-level entries and CMake leaves the nested `micromegas_<version>/`
#              directory in place.
#   * macOS  — libarchive may fold the AppleDouble sidecar into extended attributes,
#              leaving a single top-level entry that CMake then flattens.
# This script makes both cases converge on one layout: everything directly under
# <source_dir>, so the patch/build/install commands can all target <source_dir> itself.
#
# Idempotent: once the tree is flat this is a no-op, so it is safe on incremental rebuilds.
#
# Usage: normalise_micromegas_layout.sh <source_dir> <version>

set -euo pipefail

SRC="${1:?usage: normalise_micromegas_layout.sh <source_dir> <version>}"
VER="${2:?usage: normalise_micromegas_layout.sh <source_dir> <version>}"
NESTED="${SRC}/micromegas_${VER}"

# Drop the AppleDouble sidecar if tar extracted it as a real file (Linux case).
rm -f "${SRC}/._micromegas_${VER}"

if [[ -d "${NESTED}" ]]; then
    # Move everything (including dotfiles) from the nested directory up one level.
    shopt -s dotglob
    mv "${NESTED}"/* "${SRC}/"
    shopt -u dotglob
    rmdir "${NESTED}"
    echo "[normalise_micromegas_layout] flattened ${NESTED} into ${SRC}"
else
    echo "[normalise_micromegas_layout] source tree already flat; nothing to do"
fi
