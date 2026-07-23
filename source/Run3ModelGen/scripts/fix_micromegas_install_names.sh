#!/usr/bin/env bash
#
# macOS-only post-build fixup for micrOMEGAs shared libraries.
#
# micrOMEGAs builds lib/maxGap.so with a *relative* dylib id ("../lib/maxGap.so") and
# links MSSM/main against that relative path. On macOS, dyld resolves relative load
# paths against the current working directory, so `main` fails to start when it is run
# from the scan directory ("Library not loaded: ../lib/maxGap.so"). We rewrite the id
# and the reference to absolute paths, matching how the other micrOMEGAs libraries
# (e.g. sqme_aux.so) are already linked.
#
# This is a no-op on non-Darwin platforms, so it is safe to call unconditionally from
# the build.
#
# Usage: fix_micromegas_install_names.sh <micromegas_source_dir>

set -euo pipefail

MO_DIR="${1:?usage: fix_micromegas_install_names.sh <micromegas_source_dir>}"

if [[ "$(uname)" != "Darwin" ]]; then
    exit 0
fi

MAXGAP="${MO_DIR}/lib/maxGap.so"
MAIN="${MO_DIR}/MSSM/main"

if [[ ! -f "$MAXGAP" || ! -f "$MAIN" ]]; then
    echo "[fix_micromegas_install_names] WARNING: expected files not found; skipping" >&2
    echo "  maxGap.so: $MAXGAP" >&2
    echo "  main:      $MAIN" >&2
    exit 0
fi

chmod u+w "$MAXGAP" "$MAIN"

# Give maxGap.so an absolute install id ...
install_name_tool -id "$MAXGAP" "$MAXGAP"
# ... and point main at that absolute path instead of the relative one.
# (No-op if the relative reference is already absent, e.g. on an incremental rebuild.)
install_name_tool -change ../lib/maxGap.so "$MAXGAP" "$MAIN" || true

echo "[fix_micromegas_install_names] rewrote maxGap.so install names to absolute paths"
