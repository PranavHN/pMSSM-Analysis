#!/usr/bin/env bash
#
# Warm the micrOMEGAs CalcHEP process-library cache at build time.
#
# The first time micrOMEGAs' `main` evaluates a model it makes CalcHEP generate and
# compile the shared libraries for every needed matrix element (relic-density
# annihilation/coannihilation channels and the direct-detection nucleon amplitudes)
# into MSSM/work/so_generated/. These libraries are process-level, not point-level, so
# they are reused by every subsequent model and every subsequent genModels run against
# the same build. Doing this once here (on the micrOMEGAs-shipped sample spectrum) means
# the ~5 min one-time compilation is paid during `pixi run build` — and, in CI, baked
# into the Docker image — instead of on the first scan point of every job.
#
# Best-effort: the CalcHEP libraries also build on first use at run time, so a failure
# here must not fail the build. This script therefore always exits 0.
#
# The CalcHEP compilation printouts are streamed to stdout on purpose, so build logs show
# what is being compiled.
#
# Usage: warm_micromegas_cache.sh <micromegas_source_dir>

MO_DIR="${1:?usage: warm_micromegas_cache.sh <micromegas_source_dir>}"

MAIN="${MO_DIR}/MSSM/main"
SLHA="${MO_DIR}/MSSM/input.slha"
SO_DIR="${MO_DIR}/MSSM/work/so_generated"

if [[ ! -x "$MAIN" || ! -f "$SLHA" ]]; then
    echo "[warm_micromegas_cache] main or sample SLHA not found; skipping (SModelS/micrOMEGAs will compile at run time)" >&2
    echo "  main: $MAIN" >&2
    echo "  slha: $SLHA" >&2
    exit 0
fi

# Idempotent: if the process libraries are already present (e.g. an incremental rebuild),
# there is nothing to warm.
if compgen -G "${SO_DIR}/*.so" > /dev/null 2>&1; then
    echo "[warm_micromegas_cache] process libraries already present in ${SO_DIR} — nothing to do."
    exit 0
fi

echo "[warm_micromegas_cache] pre-compiling micrOMEGAs CalcHEP process libraries on the sample spectrum (${SLHA})."
echo "[warm_micromegas_cache] this is a one-time cost; CalcHEP compilation output follows:"
echo "-------------------------------------------------------------------------------"

# Run from the MSSM directory so `main` resolves its work/ tree next to the binary.
# Stream all output so the build log shows the CalcHEP compilation progress.
( cd "${MO_DIR}/MSSM" && ./main input.slha )
rc=$?

echo "-------------------------------------------------------------------------------"
if [[ $rc -eq 0 ]] && compgen -G "${SO_DIR}/*.so" > /dev/null 2>&1; then
    n=$(compgen -G "${SO_DIR}/*.so" | wc -l | tr -d ' ')
    echo "[warm_micromegas_cache] SUCCESS: cached ${n} process libraries in ${SO_DIR}."
else
    echo "[warm_micromegas_cache] WARNING: warming did not complete (exit ${rc}); libraries will be built at run time instead." >&2
fi

# Always succeed — this is a best-effort optimisation.
exit 0
