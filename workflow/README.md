# `workflow/` — layout guide

Orientation map for the Snakemake workflow **sources**: what lives in which file
and which invariants a change has to respect.

For *using* the workflow — entrypoint flags, profiles, resume semantics,
reproducibility, DAG snapshots — see **[doc/snakemake.md](../doc/snakemake.md)**;
this file does not repeat any of it.

## Directory layout

```
workflow/
├── Snakefile              config parsing, scan table, shell.prefix, rule all, DAG snapshots
├── rules/
│   ├── warmup.smk         warm_pythia8, warm_smodels_db          (localrules, always included)
│   ├── generate.smk       sample, step, ntuple                   (point-sampled priors)
│   ├── chain.smk          chain                                  (prior: mcmc)
│   ├── publish.smk        tarball, publish                       (mode=batch, publish=eos)
│   └── extract.smk        extract_batch, extract_all             (mode=extract)
├── scripts/
│   └── r3mg_batch.py      all rule bodies (internal CLI)
└── profiles/
    ├── local/config.yaml      laptop / CI / macOS
    ├── slurm/config.yaml      MPCDF viper defaults
    └── htcondor/config.yaml   CERN lxplus / lxbatch
```

Which `.smk` files get included is decided in
[Snakefile:170-178](Snakefile#L170-L178) from `mode` and the config's `prior`:

| mode | prior | included |
| - | - | - |
| `single` / `batch` | `flat`, `fixed`, `grid`, `importance` | `warmup` + `generate` + `publish` |
| `single` / `batch` | `mcmc` | `warmup` + `chain` + `publish` |
| `extract` | — | `extract` only |

`publish.smk` and `chain.smk` are always *included* but their rules are guarded
by `if MODE == ...`, so they contribute nothing to the DAG outside their mode.

## Rules

| rule | file | wildcards | declared output |
| - | - | - | - |
| `warm_pythia8` | warmup.smk | — | `build/.warm/pythia8.ok` |
| `warm_smodels_db` | warmup.smk | — | `build/.warm/smodels_db.ok` |
| `sample` | generate.smk | `scan` | `r3mg/<scan>/points.npz` |
| `step` | generate.smk | `scan`, `step` | `r3mg/<scan>/step_<step>.json` |
| `ntuple` | generate.smk | `b` (batch mode) | `<scan_dir>/ntuple.<C>.<P>.root` |
| `chain` | chain.smk | `b` (batch mode) | ntuple + `r3mg/<scan>/chain.json` |
| `tarball` | publish.smk | `b` | `scan.<runid>.<b>.tar.gz` |
| `publish` | publish.smk | `b` | `r3mg/<scan>/published.ok` (also *deletes* the scan dir + tarball unless `keep_local`) |
| `extract_batch` | extract.smk | `p` (ProcId) | `r3mg_extract/<C>.<p>.ok` |

`ntuple` and `chain` each exist in two variants (`if MODE == "single"` / `else`)
because the single-mode paths are constants while batch mode carries a `b`
wildcard — same body, different signature.

Each step manifest also records which scheduler job produced it
(`scheduler.job_id`, via `pipeline.scheduler_identity()`) — a record only,
never fed back into a name or seed. See doc/snakemake.md.

The generic `step` rule is instantiated once per entry in the resolved step list
(`prep_input`, `SPheno`, `softsusy`, `micromegas`, `superiso`, `gm2calc`,
`feynhiggs`, `SModelS` in the default config), chained through the manifests by
[`_step_inputs`](rules/generate.smk#L10-L18) in config order.

## `scripts/r3mg_batch.py`

Every rule body. A click group; one subcommand = one rule for one batch. Invoked
by absolute path from the rules (`{RULE_SCRIPTS}/r3mg_batch.py ...`), deliberately
not on `PATH` — it is not a user entrypoint.

| subcommand | required options | does |
| - | - | - |
| `sample` | `--scan_dir --seed --points_out` | wipes + recreates the scan dir, samples all points, dumps `scan_config.yaml` and the points `.npz` |
| `step` | `--step --scan_dir --points --manifest_out` (`--prev_manifest`) | runs one tool over the models gated by the previous manifest, writes this step's manifest |
| `ntuple` | `--scan_dir --points` | re-prints the per-step summary, then `mkntuple()` |
| `chain` | `--scan_dir --seed --manifest_out` | one whole MH scan as a single opaque job |
| `tarball` | `--scan_dir --tar_out` | hardlink-stages the tree, folds `SModelS_xsec/` back over `SPheno/`, tars |
| `publish` | `--scan_dir --tarball --runid --procid` (`--cleanup`) | `cp` (or `xrdcp` for `root://`) to `$EOSPATH/<runid>/<procid>/`; with `--cleanup`, verifies every destination's size and then removes the local scan dir + tarball |

`--config_file` is optional everywhere (absent = default config).

Every step manifest (and `chain.json`) carries a `scheduler` block recording
which batch job wrote it — `{backend, job_id, host, ...}` from
`pipeline.scheduler_identity()`, read from `SLURM_JOB_ID` or HTCondor's
`_CONDOR_JOB_AD`. It exists so "batch 17 failed" maps straight to a scheduler
job id: otherwise that mapping lives only in the orchestrator log, and only as
ClusterId ↔ group UUID, so recovering the batch index means recomputing
`uuid5("publish:b:17")`. It is a **record, not an identifier** — nothing derives
a name, path or seed from it, deliberately (see `pipeline.batch_seed`).

## Invariants

Four constraints that are easy to break and expensive to debug:

1. **Parse-time purity.** `Snakefile` and the `.smk` files import only the stdlib
   and `yaml` ([Snakefile:29-32](Snakefile#L29-L32)). Remote executors re-parse
   the Snakefile inside every job, where the `Run3ModelGen` package is not yet
   importable. Anything heavier belongs in `r3mg_batch.py`, which runs after
   `build/setup.sh` has been sourced by the [shell prefix](Snakefile#L58-L59).

2. **Parse-time mirrors of runtime logic.** `_load_cfg`, `_resolve_steps`,
   `_batch_seed` and `_ntuple_basename` in the Snakefile duplicate
   `pipeline.load_scan_config` / `resolve_steps` / `batch_seed` / `ntuple_name`
   as a consequence of invariant 1. **Change both sides together** — a drift
   silently produces a DAG that disagrees with what the rule bodies do.
   [tests/test_snakefile_mirrors.py](../tests/test_snakefile_mirrors.py) pulls
   the four mirrors out of the Snakefile as source text and compares them
   against `pipeline.py`, so a drift (or a rename) fails the suite instead.

3. **Exit code = infrastructure, manifest = physics.** A rule body exits non-zero
   only on a broken environment or missing files; per-model tool failures are
   recorded in the manifest with exit 0. Otherwise snakemake `retries` would mask
   physics failures and the CI per-step reference counts would move.

4. **A deleted output must not be a requested target.** Publishing removes each
   batch's scan dir and tarball, so with `keep_local` off
   [`_final_targets`](Snakefile#L181-L199) returns *only* the `published.ok`
   markers. Snakemake leaves a missing **intermediate** alone but regenerates a
   missing **requested** file — listing the ntuples there would re-run every
   batch from scratch to re-create exactly what publish just deleted.

5. **Bookkeeping stays outside the scan dirs.** Manifests and markers live under
   `r3mg/` so tarball contents stay byte-identical to the legacy layout. Per-model
   files inside the scan dir are *not* declared outputs (failed models legitimately
   have none) — hence the explicit out-of-sync guards in `r3mg_batch.py`'s `step`
   and `ntuple`.

6. **Bookkeeping splits by lifetime.** Those guards are why there are two roots:
   `WORK` (`<scan_parent>/r3mg/`) holds the per-batch manifests and `META`
   (`<meta_parent>/r3mg/`) the `published.ok` markers, run metadata and DAG
   snapshot. The same directory unless `meta_parent` is set. A manifest describes
   the contents of its scan dir, so it must not outlive it — otherwise an evicted
   batch whose data is gone resumes into the out-of-sync guard and needs a manual
   `rm -rf`. A marker must outlive the job, because the submit-side orchestrator
   reads it to decide what is finished. That is what lets `genModels.py --backend
   htcondor` put the batch data in each worker's `$_CONDOR_SCRATCH_DIR`.

## Environment

There are no per-rule `conda:` or `container:` directives. Every rule shell gets
the repo's prebuilt pixi env by absolute path plus `build/setup.sh`, via the
single `shell.prefix` at [Snakefile:58-59](Snakefile#L58-L59); this assumes the
checkout and its `.pixi/` sit on a filesystem the workers can see. See
[doc/pixi.md](../doc/pixi.md).

## Editing checklist

- Adding a rule → put it in the `.smk` matching its mode, and check whether the
  cluster profiles need it in their `groups:` list (all per-batch rules are
  grouped so one batch = one cluster job).
- Touching config keys → update the `Snakefile` header docstring, which is the
  canonical key reference.
- Any change → `pixi run test` runs the unit tests plus the workflow DAG dry-runs
  (`tests/run_checks.sh`, the same script CI uses).
