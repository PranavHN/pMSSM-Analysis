#!/usr/bin/env bash
#
# setup_pythia8.sh — ensure SModelS's bundled pythia8 is built.
#
# Workaround for two upstream SModelS bugs (seen with smodels 3.1.1) that leave
# pythia8 unbuilt and make genModels.py's SModelS step fail with the misleading
# "Compilation of pythia8 failed. Is the C++ compiler installed?":
#
#   1. installer.py's primary download URL (https://pythia.org/download/...) now
#      returns HTTP 404, so it always falls back to smodels.github.io (GitHub
#      Pages / Fastly). Because python-requests advertises "Accept-Encoding: gzip",
#      the CDN wraps the already-.tgz body in an extra "Content-Encoding: gzip",
#      and installer.py saves the *undecoded* stream via
#      shutil.copyfileobj(r.raw, ...) -> a DOUBLE-gzipped tarball that
#      tarfile.open(..,'r:gz') cannot extract -> pythia8 is never built.
#   2. installer.py won't re-download a tarball already >15 MB, so the corrupt
#      file is reused on every run.
#
# This script detects/repairs a double-gzipped tarball (or fetches a clean one
# from the working /releases/ URL with proper decoding) and builds pythia8. It is:
#   - idempotent  : exits immediately if pythia8 is already built,
#   - locked      : flock serializes concurrent invocations (parallel condor/slurm
#                   jobs, or a build racing a manual run) so they can't corrupt
#                   the shared install,
#   - polite      : caps parallelism via taskset (SModelS otherwise compiles with
#                   nCPUs()-2 jobs, e.g. 254 on a 256-core node),
#   - self-effacing: if SModelS is fixed/restructured upstream it steps aside and
#                   delegates to SModelS's own build.
#
set -euo pipefail

# --- locate the installed smodels pythia8 dir --------------------------------
PYBIN="$(command -v python || command -v python3 || true)"
[[ -n "$PYBIN" ]] || { echo "[setup_pythia8] no python found; skipping."; exit 0; }
PYDIR="$("$PYBIN" -c 'import os, smodels; print(os.path.join(os.path.dirname(smodels.__file__), "lib", "pythia8"))' 2>/dev/null || true)"

# Step aside gracefully if SModelS isn't laid out the way this workaround expects
# (e.g. a fixed upstream ships pythia8 differently) — never block the build.
if [[ -z "$PYDIR" || ! -d "$PYDIR" ]]; then
  echo "[setup_pythia8] smodels pythia8 dir not found; assuming SModelS handles pythia8 itself. Skipping."
  exit 0
fi
cd "$PYDIR"
if [[ ! -f pythiaversion || ! -f Makefile || ! -f installer.py ]]; then
  echo "[setup_pythia8] unexpected pythia8 layout (no pythiaversion/Makefile/installer.py); SModelS may build pythia8 differently now. Skipping."
  exit 0
fi
VER="$(tr -d '[:space:]' < pythiaversion)"      # e.g. 8317
TARBALL="pythia${VER}.tgz"

# --- serialize: never let two builds run in this dir at once -----------------
# (skip locking gracefully if flock is unavailable, rather than aborting the build)
if command -v flock >/dev/null 2>&1; then
  exec 9>".setup_pythia8.lock"
  if ! flock -w 1200 9; then
    echo "[setup_pythia8] lock busy after 1200s; another build is running. Skipping." >&2
    exit 0
  fi
fi

# already built? (re-checked here now that we hold the lock)
if [[ -x pythia8.exe && -f "pythia${VER}/lib/libpythia8.a" ]]; then
  echo "[setup_pythia8] pythia8 already built in ${PYDIR} — nothing to do."
  exit 0
fi

echo "[setup_pythia8] building pythia8 in ${PYDIR}"
# drop any partial/broken extraction so the rebuild is deterministic
rm -rf "pythia${VER}"

# --- ensure a valid tarball --------------------------------------------------
if [[ -f "$TARBALL" ]] && ! tar -tzf "$TARBALL" >/dev/null 2>&1; then
  if zcat "$TARBALL" 2>/dev/null | tar -tz >/dev/null 2>&1; then
    echo "[setup_pythia8] ${TARBALL} is double-gzipped; stripping one gzip layer."
    zcat "$TARBALL" > "${TARBALL}.fixed" && mv -f "${TARBALL}.fixed" "$TARBALL"
  else
    echo "[setup_pythia8] ${TARBALL} is corrupt and unrecognized; re-downloading."
    rm -f "$TARBALL"
  fi
fi
if [[ ! -f "$TARBALL" ]]; then
  # /releases/ is the live path (302 -> pythia8.web.cern.ch); --compressed makes
  # curl correctly decode any Content-Encoding: gzip a CDN might add.
  for u in "https://pythia.org/releases/pythia${VER:0:2}/${TARBALL}" \
           "https://smodels.github.io/downloads/tarballs/${TARBALL}"; do
    echo "[setup_pythia8] downloading ${u}"
    if curl -fSL --compressed "$u" -o "${TARBALL}.dl" \
       && tar -tzf "${TARBALL}.dl" >/dev/null 2>&1; then
      mv -f "${TARBALL}.dl" "$TARBALL"
      break
    fi
    echo "[setup_pythia8] fetch from ${u} failed or was not a valid tarball."
    rm -f "${TARBALL}.dl"
  done
fi
[[ -f "$TARBALL" ]] || { echo "[setup_pythia8] ERROR: could not obtain a valid ${TARBALL}" >&2; exit 1; }

# --- build ------------------------------------------------------------------
# The bundled Makefile's `all: install_pythia8 pythia8.exe` declares NO ordering
# between the two prerequisites. Under a parallel top-level make — e.g. the
# jobserver inherited through MAKEFLAGS from `cmake --build --parallel N` — the
# pythia8.exe link runs concurrently with (and before) libpythia8.a finishes
# building, which fails the build. (This is what broke CI.) So: drop any inherited
# jobserver, then build the library first and link second, as two serial steps.
unset MAKEFLAGS MFLAGS MAKELEVEL

# Cap the library's internal parallelism: installer.py compiles it with
# `make -j (nCPUs()-2)`, and nCPUs() reads the CPU-affinity mask, so taskset caps
# it. Cap by cores AND memory (~2 GB per job; pythia8's big TUs are memory-hungry)
# so we also stay within budget on small runners.
CAP="${PYTHIA8_BUILD_NCORES:-8}"
mem_gb=$(awk '/^MemTotal:/{print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [[ "$mem_gb" -gt 0 ]]; then
  mem_cap=$(( mem_gb / 2 )); [[ "$mem_cap" -lt 1 ]] && mem_cap=1
  [[ "$mem_cap" -lt "$CAP" ]] && CAP="$mem_cap"
fi
TS=""
if command -v taskset >/dev/null 2>&1 && [[ "$(nproc)" -gt "$CAP" ]]; then
  TS="taskset -c 0-$((CAP-1))"
fi
echo "[setup_pythia8] compiling with up to ${CAP} cores (mem=${mem_gb}GB)…"
$TS make -j1 install_pythia8   # installer.py: extract + build libpythia8.a (its own make -j)
$TS make -j1 pythia8.exe       # then link the driver against the finished library

if [[ -x pythia8.exe && -f "pythia${VER}/lib/libpythia8.a" ]]; then
  echo "[setup_pythia8] SUCCESS: built pythia8.exe and libpythia8.a in ${PYDIR}"
else
  echo "[setup_pythia8] ERROR: build did not produce pythia8.exe / libpythia8.a" >&2
  exit 1
fi
