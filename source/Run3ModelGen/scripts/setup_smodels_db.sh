#!/usr/bin/env bash
#
# setup_smodels_db.sh — pre-fetch the SModelS "official" experimental-results
# database pickle into the SModelS cache directory.
#
# genModels.py's SModelS step loads the database with `path = official` (see
# data/SModelS_parameters.ini), which makes SModelS download a large binary pickle
# (official<ver>.pcl, hundreds of MB) into cacheDirectory() on first use. SModelS
# only reuses a cached pickle when it already lives in that cache dir (it compares
# size / lastchanged, and even falls back to the local copy when the server is
# unreachable). The cache dir is $SMODELS_CACHEDIR if set, else ~/.cache/smodels.
#
# In CI each job starts from a fresh checkout / fresh pixi env, and neither
# ~/.cache nor the repo tree is carried between jobs by default, so the database is
# re-downloaded on the first SModelS run of *every* genModels job. This script
# downloads it ONCE at build time; CI then points $SMODELS_CACHEDIR at a directory
# that IS carried between jobs (a GitLab artifact for the local track, an
# image-baked path for the docker track — see .gitlab-ci.yml and Dockerfile), so
# downstream jobs reuse the prefetched pickle instead of re-downloading it.
#
# It is:
#   - idempotent  : exits immediately if a database pickle is already cached,
#   - self-effacing: skips quietly if python/smodels aren't available or laid out
#                    as expected, rather than blocking the build.
#
set -euo pipefail

# Best-effort mode (used by the CMake build target): the database is only needed at
# *run* time, so a failed prefetch must NOT abort the whole build — turn any nonzero
# exit into a warning + exit 0 and let SModelS download it on first use instead.
if [[ -n "${SETUP_SMODELS_DB_BEST_EFFORT:-}" ]]; then
  trap 'rc=$?; [[ $rc -eq 0 ]] || echo "[setup_smodels_db] WARNING: prefetch failed (exit $rc); SModelS will download the database at first use."; exit 0' EXIT
fi

PYBIN="$(command -v python || command -v python3 || true)"
[[ -n "$PYBIN" ]] || { echo "[setup_smodels_db] no python found; skipping."; exit 0; }

# Step aside gracefully if smodels isn't importable (e.g. env not built yet).
if ! "$PYBIN" -c 'import smodels' >/dev/null 2>&1; then
  echo "[setup_smodels_db] smodels not importable; skipping (SModelS will fetch the database at run time)."
  exit 0
fi

# Resolve the cache dir exactly the way SModelS does (honours $SMODELS_CACHEDIR,
# else ~/.cache/smodels), so the prefetch lands where the SModelS run will look.
CACHE_DIR="$("$PYBIN" -c 'from smodels.installation import cacheDirectory; print(cacheDirectory(create=True))' 2>/dev/null || true)"
if [[ -z "$CACHE_DIR" ]]; then
  echo "[setup_smodels_db] could not resolve SModelS cache dir; skipping."
  exit 0
fi
echo "[setup_smodels_db] SModelS database cache dir: ${CACHE_DIR}"

# Idempotent: if a database pickle is already cached, there is nothing to do (and we
# avoid an unnecessary network round-trip on incremental rebuilds).
if compgen -G "${CACHE_DIR}/*.pcl" >/dev/null 2>&1; then
  echo "[setup_smodels_db] database pickle already cached in ${CACHE_DIR} — nothing to do."
  exit 0
fi

echo "[setup_smodels_db] fetching the official SModelS database (one-time download)…"
"$PYBIN" - <<'PY'
from smodels.experiment.databaseObj import Database
# Constructing the database with the "official" label downloads the version-matched
# pickle into the cache dir and loads it, warming the cache for later runs.
db = Database("official")
try:
    print(f"[setup_smodels_db] database loaded: version {db.databaseVersion}")
except Exception:
    print("[setup_smodels_db] database loaded.")
PY

if compgen -G "${CACHE_DIR}/*.pcl" >/dev/null 2>&1; then
  f="$(compgen -G "${CACHE_DIR}/*.pcl" | head -1)"
  echo "[setup_smodels_db] SUCCESS: cached $(basename "$f") in ${CACHE_DIR}"
else
  echo "[setup_smodels_db] ERROR: no database pickle present after fetch." >&2
  exit 1
fi
