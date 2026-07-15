# SModelS pythia8 auto-build

SModelS computes LO cross sections with a bundled copy of pythia8 that it compiles
on first use. In this project that auto-build is broken upstream, so Run3ModelGen
builds it explicitly. This note explains what happens and how to control it.

## Why it's needed

SModelS (3.1.1) fetches the pythia8 source itself, but:

1. Its primary download URL (`https://pythia.org/download/pythia8X/...`) now returns
   HTTP 404, so it falls back to `smodels.github.io` (a Fastly CDN).
2. The CDN serves the already-gzipped `.tgz` with an extra `Content-Encoding: gzip`,
   and SModelS's `installer.py` saves the *undecoded* stream — producing a
   **double-gzipped** tarball that `tarfile` cannot extract.
3. `installer.py` won't re-download a tarball larger than 15 MB, so the corrupt file
   is reused on every run.

The net effect is the misleading error `Compilation of pythia8 failed. Is the C++
compiler installed?` even though the compiler is fine.

## How Run3ModelGen fixes it

[`source/Run3ModelGen/scripts/setup_pythia8.sh`](../source/Run3ModelGen/scripts/setup_pythia8.sh)
locates the installed SModelS, repairs a double-gzipped tarball (or downloads a clean
one from the working `/releases/` URL with `curl --compressed`), and builds pythia8. It is:

- **idempotent** — a fast no-op once `pythia8.exe`/`libpythia8.a` exist;
- **locked** (`flock`) so concurrent jobs can't corrupt the shared install;
- **memory-capped** (`taskset`) so it doesn't oversubscribe a small runner, and it
  builds the library and links the executable as two serial steps (the bundled
  Makefile has no ordering between them, which races under a parallel `make`);
- **self-effacing** — if a future SModelS ships pythia8 differently it steps aside.

It is invoked from two places:

1. **The CMake build** — an `add_custom_target(setup_pythia8 ALL ...)` in
   [`source/CMakeLists.txt`](../source/CMakeLists.txt) builds pythia8 during
   `pixi run build` (best-effort / non-fatal — pythia8 is only needed at run time).
2. **`build/setup.sh`** — [`source/cmake_build_env.sh`](../source/cmake_build_env.sh)
   appends a call so every `source build/setup.sh` ensures pythia8. This covers jobs
   that start from a fresh pixi environment where the build-time copy didn't persist
   (e.g. CI `genModels` jobs, which re-create `.pixi`).

## Manual control

- Build/repair it by hand: `pixi run setup-pythia8`, or
  `pixi run bash source/Run3ModelGen/scripts/setup_pythia8.sh`.
- Cap the compile cores: `PYTHIA8_BUILD_NCORES=4 pixi run setup-pythia8`.
- Force a clean rebuild: delete `pythia8.exe` and the `pythia<ver>/` directory under
  `.pixi/envs/default/lib/python3.11/site-packages/smodels/lib/pythia8/`, then re-run.
