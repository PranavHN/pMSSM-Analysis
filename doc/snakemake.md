# Snakemake workflow

Run3ModelGen's execution is orchestrated by [Snakemake](https://snakemake.readthedocs.io)
(`workflow/Snakefile`). The repo's scripts are the entrypoints — backend-independent
via a shared `--backend local|slurm|htcondor` flag that selects the executor profile:

| script | drives | modes |
| - | - | - |
| `genModels.py` | one scan (`--num_jobs 1`, fresh-scan overwrite) or N batches (`--num_jobs N`, resume-friendly, optional `--publish`) | single / batch |
| `extractModels.py` | one scan dir/tarball in-process (no snakemake), or per-ProcId fan-out over a published run (`--eos_dir`) | single / extract |
| `rerunStep.py` | re-run one step (+ downstream) of an existing scan | single |

> This document is the **user guide**. For the layout of the workflow sources —
> which rule lives in which `.smk`, the `r3mg_batch.py` subcommand interface, and
> the invariants a change must respect — see
> [workflow/README.md](../workflow/README.md).

Power users can drive the workflow directly; the `smk` pixi task wraps the
environment setup and runs from `run/` (all arguments pass through). This is
also the escape hatch for shapes the scripts don't expose (e.g. a published
batch-of-1: `mode=batch batches=1 publish=eos`):

```bash
pixi run smk --workflow-profile ../workflow/profiles/local \
    --config mode=single scan_dir=scan config_file=myconfig.yaml seed=123
# equivalently, with the environment already set up:
source build/setup.sh
snakemake -s workflow/Snakefile --workflow-profile workflow/profiles/local --config ...
```

## Entrypoint reference

Shared conventions: flags left at their defaults never conflict; explicitly
passed flags that cannot take effect are **hard errors, raised before any side
effect** (no wipe, no submission). Cluster backends (`slurm`/`htcondor`)
detach the snakemake orchestrator by default (`--foreground` to attach; the
log path is printed at submission). `local` always runs in the foreground.

### genModels.py

| flag | default | applies to | meaning |
| - | - | - | - |
| `--config_file` | default config | always | scan config yaml (validated before anything is wiped) |
| `--scan_dir` | `scan` (htcondor batch: each worker's `$_CONDOR_SCRATCH_DIR/scan`) | always | the scan dir (`--num_jobs 1`) or the parent of the `scan.<runid>.<b>` batch dirs (`--num_jobs > 1`) — see [Batch data on the worker](#batch-data-on-the-worker) |
| `--seed` | 123 | single | numpy seed, used as-is |
| `--backend` | `local` | always | `local` \| `slurm` \| `htcondor` |
| `--num_jobs` | 1 | always | 1 → single mode (fresh-scan overwrite); N → batch mode (never wipes, resumable) |
| `--runid` | timestamp | batch | digits-only run identifier (naming + seeds + EOS layout) |
| `--base_seed` | 123 | batch | per-batch seed = `blake2b(runid:base_seed:b)` |
| `--publish` | off | batch | tar every batch and **move** tarball/ntuple/config to `$EOSPATH/<runid>/<b>/` (any backend; `$EOSPATH` must be writable or `root://`) — see [Publishing is a move](#publishing-is-a-move) |
| `--keep_local` | off | batch + publish | keep every batch's scan dir and tarball on the shared filesystem as well |
| `--flavour` | `tomorrow` | htcondor | JobFlavour (`espresso` for tests) |
| `--log_dir` | under `r3mg/` | cluster | orchestrator log + per-job scheduler logs (`outfiles/`) + config snapshot |
| `--foreground` | off | cluster | stay attached; Ctrl-C then cancels the cluster jobs |

Batch resume: **re-run the exact same command** — finished batches are skipped.
"Exact" is load-bearing: the flags decide which files snakemake is asked to
produce, so a different set is a different question, not a gentler one. Dropping
`--publish` from a published run asks for ntuples that publishing deliberately
deleted; that is refused up front (add `--publish` back, or use a new `--runid`).
Changing the config content or `--base_seed` is refused too, since it would
re-sample. If you only want to know whether a run finished, re-running is safe —
it rebuilds just what is missing and exits 0 with nothing to do otherwise.
The per-run manifest `<scan_parent>/r3mg/run.<runid>.yaml` refuses re-submission
with changed config content or `--base_seed` (that would re-sample and wipe
finished batches — use a new runid), warns on `--num_jobs` top-ups (new batches
get their deterministic seeds), allows adding `--publish` later, and blocks a
second submission while the recorded orchestrator is still alive.

### extractModels.py

Without `--eos_dir`: the historical single-scan extraction, in-process
(`--scan_dir` dir **or** tarball + `--root_file` + `--selection`), no snakemake
involved. With `--eos_dir`: one extraction job per ProcId subdirectory of a
published run:

| flag | default | meaning |
| - | - | - |
| `--eos_dir` | — | `$EOSPATH/<runid>`-style run directory (must be listable from this host) |
| `--backend` / `--flavour` / `--log_dir` / `--foreground` | as genModels.py | same semantics; `--log_dir` also holds the per-ProcId completion markers, so re-runs skip finished ProcIds on every backend |
| `--num_jobs` | all ProcId subdirs | explicit N expects dirs `0..N-1` to exist (fail-fast) |
| `--selection` | `akarr['SS_m_h']!=-1` | awkward-array mask, both modes |

### rerunStep.py

| flag | default | meaning |
| - | - | - |
| `--step` | — | pipeline step to re-run (or `ntuple` for just the ntuple) |
| `--scan_dir` | `scan` | existing scan |
| `--config_file` / `--seed` | recovered from `scan_config.yaml` | explicit overrides for unusual cases |
| `--backend` / `--log_dir` / `--foreground` | as genModels.py | cluster backends submit the tail as one grouped job |

rerunStep pins snakemake to mtime-only rerun triggers: code/config drift since
the original run can therefore never mark the `sample` rule outdated (which
would wipe the scan). It also refuses to start while another snakemake run is
active from the same directory.

## Rule graph

For the point-sampled priors (`flat`, `fixed`, `grid`, `importance`), one batch =
one scan directory, and every physics tool is its own rule:

```
warm_pythia8 / warm_smodels_db   (localrules, once per environment)
sample  →  prep_input → SPheno → softsusy → micromegas → superiso
        →  gm2calc → feynhiggs → SModelS  →  ntuple  [→ tarball → publish]
```

- **Re-running a single step**: `rerunStep.py --step micromegas --scan_dir scan`
  re-runs micromegas + everything after it (nothing before it) and rebuilds the
  ntuple; original config/seed are recovered from the scan's `scan_config.yaml`.
  (Manually you must delete the manifests of the step AND everything downstream
  plus the ntuple — snakemake does not re-validate finished jobs against a
  missing mid-chain manifest, so deleting only one is not enough.)
- **Per-model tool failure is data, not an error** (random pMSSM points fail often):
  a rule exits non-zero only on infrastructure problems, so snakemake `retries`
  never mask physics failures. Failed models simply lack output files and get the
  usual `-1` ntuple sentinels.
- The tool rules are chained **in config-step order** (not by data dependency):
  a model that fails softsusy is also skipped for micromegas, exactly like the
  legacy in-process loop — this keeps the per-step success counts compared by the
  CI reference check byte-identical.
- `prior: mcmc` instead runs as **one opaque `chain` rule per batch** (the
  Metropolis-Hastings chain runs the whole pipeline inside sampling and cannot be
  decomposed). This rule is also the seam where the emcee sampler from the `mcmc`
  branch will plug in (engine dispatch inside `r3mg_batch.py chain`, plus
  `chain.h5` resume state).

Rule bodies live in `workflow/scripts/r3mg_batch.py` (internal CLI:
`sample|step|ntuple|chain|tarball|publish`; invoked by path from the rules, never
by hand — it is deliberately not on `PATH`);
every rule shell first sources `build/setup.sh`, reproducing the legacy condor
worker environment on all executors. Bookkeeping (per-step manifests, markers)
lives under `r3mg/`, **outside** the scan dirs, so tarball contents are unchanged —
in one root by default, split into `<scan_parent>/r3mg/` (manifests) and
`<meta_parent>/r3mg/` (markers, run metadata) when the scan dirs are ephemeral; see
[Publishing is a move](#publishing-is-a-move).

## SModelS cross sections (immutable SPheno outputs)

Legacy code appended the computed cross sections **into** `SPheno/<n>.slha` in
place. Under Snakemake that would mutate a finished rule's output, so the
workflow writes the xsec-augmented file to `SModelS_xsec/<n>.slha` instead and
`SPheno/` stays pristine. The `tarball` rule folds the augmented copies back
over the `SPheno/` member paths (and drops `SModelS_xsec/`), so published
tarballs — what the pMSSMFactory consumes — are byte-compatible with the legacy
layout, XSECTION blocks included.

## Which batch job ran a batch

Each step manifest carries a `scheduler` block recording the job that produced
it — `backend` plus the scheduler's own id (`job_id` is `<ClusterId>.<ProcId>`
on htcondor, `SLURM_JOB_ID` on slurm, with `restart_count` added when a requeue
happened), and the worker hostname:

```console
$ jq .scheduler <scan_parent>/r3mg/scan.<runid>.14/step_SPheno.json
{ "host": "b9p12p3106.cern.ch", "backend": "htcondor", "job_id": "20423535.0",
  "global_job_id": "bigbird25.cern.ch#20423535.0#1785238088" }
```

So "batch 14's SPheno looks wrong" maps straight to a scheduler job — no
grepping the orchestrator log, which only records `ClusterId <-> group UUID`
(recovering the batch index from that means recomputing
`uuid5("publish:b:14")`). On a resume the steps that re-ran carry the new job's
id while the others keep the original, which is what shows you *which attempt*
produced what.

It is a **record, not an identifier**: nothing derives a name, path or seed from
it. The legacy scheme did exactly that (`${ClusterId:6}${ProcId}`) and it is why
runs were unreproducible — a ClusterId is unknowable before submission and
changes on every resubmission (see [Batch identity](#batch-identity--reproducibility)).
Collection never fails a rule: an unreadable job ad degrades to `backend: local`.

## Publishing is a move

`--publish` leaves the run **on EOS only**: once a batch's tarball, ntuple and
config are copied to `$EOSPATH/<runid>/<b>/`, the `publish` rule deletes that
batch's scan directory and tarball. This is the legacy disk profile — the condor
worker's scratch dir vanished with the job, and only the published copies
survived — and it is what makes large runs possible at all: one 10,000-model
batch is ~74,000 files / 1.1 GB, so a 300-batch production run would otherwise
leave ~340 GB of scan dirs plus ~75 GB of tarballs on the shared filesystem
(an AFS work volume is 100 GB).

The delete is guarded. Every published file is re-checked for presence **and
byte size** first (`xrdfs stat` for `root://` destinations, a plain stat
otherwise), and the check fails closed: if any destination is missing, short, or
unreadable, nothing is deleted and the rule fails with the mismatch listed. The
local copies are therefore only ever removed once EOS demonstrably has them.

What survives is the `r3mg/` bookkeeping — the per-step manifests, the sampled
points and the `published.ok` markers, a few hundred KB per batch. That is what
keeps a published run **resumable and inspectable**: re-running the same
`genModels.py` command is a no-op rather than a full regeneration, and the
per-step success counts stay readable. (Mechanically: with publishing on, the
ntuples drop out of the workflow's target list, because snakemake regenerates a
missing *requested* file while leaving a missing *intermediate* alone.)

That bookkeeping has **two roots**, because its pieces have different lifetimes:

| root | config key | holds | lifetime |
| - | - | - | - |
| `WORK` | `<scan_parent>/r3mg/` | per-batch manifests: `points.npz`, `step_*.json`, `chain.json` | travels with the scan data |
| `META` | `<meta_parent>/r3mg/` | `published.ok`, `run.<runid>.yaml`, `dag.<runid>.*` | outlives the job |

They are the **same directory** unless `meta_parent` is set, so a shared-filesystem
run is unchanged. The split exists for ephemeral scan dirs (next section): a
manifest describes the *contents* of its scan dir, and the `step`/`ntuple` bodies
refuse to run against a manifest whose files have vanished. Keeping manifests next
to the data means an evicted batch simply re-runs whole; leaving them on shared
storage would make it resume straight into that out-of-sync guard and demand a
manual `rm -rf`. The markers go the other way: the orchestrator on the submit host
reads them to decide what is finished, so they must not live on a worker.

`--keep_local` opts out and keeps both copies — useful for debugging, for
`rerunStep.py` on a published batch, or when the scan dirs are the product and
EOS is just a backup. Budget the disk for the whole run before using it.

Extraction is unaffected: `extractModels.py --eos_dir $EOSPATH/<runid>` reads the
published tarballs, which is the normal way to consume a published run.

## Batch data on the worker

On `--backend htcondor`, batch runs write each batch into **its own worker's
`$_CONDOR_SCRATCH_DIR/scan`** rather than under `--scan_dir`. Only the published
copies leave the node, so N concurrent batches cost the submit volume nothing
beyond the `META` bookkeeping.

This is not a tuning knob, it is the difference between a wide run working and
not. With the batch data on shared storage, peak footprint is *all batches at
once* — the htcondor profile submits every batch immediately rather than in waves
(`jobs: 10000`), so a 300-batch run needs ~340 GB of scan dirs simultaneously. A
real 300-batch run died exactly there: the submit volume hit its quota within the
hour, every job failed in HTCondor's output transfer with `(errno 122) Disk quota
exceeded`, all 300 were held, and `SYSTEM_PERIODIC_REMOVE` reaped them 24 h later.
Nothing was published; 300 batches of compute produced nothing.

Consequences worth knowing:

- **`--publish` is required.** Scratch is destroyed when the job exits, so a run
  whose results never reach EOS produces nothing. `genModels.py` refuses the
  combination up front rather than spending the cluster time.
- **`--keep_local` is refused** for the same reason: there is nothing to keep, and
  it would put the ntuples back in the target list at paths that only ever exist
  on a worker, which the orchestrator would then wait for forever.
- **Eviction costs the whole batch.** The work is on the node, so an evicted batch
  restarts from sampling rather than resuming mid-pipeline. It leaves nothing
  behind and re-runs cleanly on the next `genModels.py` invocation.
- **The disk request must cover a batch.** `htcondor_request_disk_mb` is 4000 in
  the htcondor profile (and repeated in `snakefacade.flavour_resources`, which
  *replaces* the profile block). Under-requesting does not fail cleanly — the job
  fills its scratch and dies mid-batch.

Pass `--scan_dir` explicitly to opt back into the shared-filesystem layout;
`genModels.py` then warns with the projected peak for that run. The `slurm` and
`local` backends are unaffected — only HTCondor hands each job a private scratch
directory that it then deletes.

The mechanism: `scan_parent` reaches snakemake as the **unexpanded** string
`$_CONDOR_SCRATCH_DIR/scan`, because HTCondor only defines that variable inside a
job. The Snakefile runs `os.path.expandvars` over it, so the submit-side
orchestrator keeps the literal (it never touches those paths — its only batch
targets are the `published.ok` markers) while each worker, re-parsing the Snakefile
inside its own job, resolves it to that job's scratch directory.

## Batch identity & reproducibility

`runid` (digits-only, default: timestamp) replaces the condor ClusterId in the
`scan.<runid>.<b>` / `ntuple.<runid>.<b>.root` / `$EOSPATH/<runid>/<b>/` naming,
so the model-id contract (`id = b*100000 + index`) and extractModels keep working.
Unlike the legacy `${ClusterId:6}${ProcId}` seed (unknowable before submission),
every batch seed is now `blake2b(runid:base_seed:b)` — **(runid, base_seed, config)
reproducibly determine every sampled point** of a whole run. Single mode
(`genModels.py --seed N`) uses the seed as-is, bit-compatible with the legacy CLI —
verifiable at any time with `pixi run test`, whose
`tests/test_sampling_fixture.py` asserts every sampled point against a golden
file frozen from the pre-refactor code
(`tests/fixtures/sampling_seed123.json`). After an *intentional* sampling
change, regenerate it with `tests/regen_sampling_fixture.py` and re-seed the
EOS references (`make-refs`, `upload-refs`) in the same commit — the two
contracts have to move together.

## Visualizing the DAG

**Every (non-dry) run snapshots its own DAG automatically.** At the end of the
run each scan dir receives a copy (`<scan_dir>/dag.{dot,svg}`); the snapshot is
also kept next to the bookkeeping (`r3mg/<scan>/dag.*` in single mode,
`r3mg/dag.<runid>.*` in batch mode, `r3mg_extract/dag.<ClusterId>.*` for
extraction), which survives even if the run dies mid-way (scan dirs are wiped
at the start of a fresh run, so they only get the copy at the end). The `.dot`
source is always written; the `.svg` additionally requires a graphviz `dot`
binary (PATH or the installed `dagviz` pixi env, which `pixi run build`
installs — standalone: `pixi install -e dagviz`). `dag.*` is excluded from published tarballs (legacy member layout);
snapshot failures never fail the run.

graphviz lives in its own pixi environment (`dagviz`) — pulling it into the
default environment would downgrade xrootd via its GUI dependency stack. To
render a DAG manually, pipe snakemake's dot output through it (use the same
`--config` as the run you want to inspect; `--forceall` draws the complete
graph even when outputs exist):

```bash
source build/setup.sh && cd run
# one node per (rule, wildcards) job — per-tool steps shown individually:
snakemake -s ../workflow/Snakefile --workflow-profile ../workflow/profiles/local \
    --config mode=batch runid=42 batches=2 scan_parent=. publish=eos \
    --forceall --dag | pixi run -e dagviz dot -Tsvg > dag.svg
# condensed one-node-per-rule overview: swap --dag for --rulegraph
```

Note the generic `step` rule appears as a single node in `--rulegraph`; use
`--dag` to see the eight tool instances spelled out per batch.

## Profiles

- `workflow/profiles/local` — laptop/CI/macOS (on macOS the feynhiggs rule drops
  out of the DAG automatically, matching the legacy runtime behavior).
- `workflow/profiles/slurm` — native SLURM submission. One cluster job per batch
  (all per-batch rules share a snakemake *group*), `slurm-requeue`, retries.
  Site defaults target viper (`general` partition; put `scan_parent` on `/ptmp`).
- `workflow/profiles/htcondor` — native HTCondor submission (CERN lxplus; shared-FS
  checkout and `.pixi` env, but the batch DATA goes to per-job scratch by default —
  see [Batch data on the worker](#batch-data-on-the-worker)).
  JobFlavour/MaxRuntime/RequestDisk/RequestCpus map to
  `classad_*`/`htcondor_*` resources; `RequestDisk` is no longer the historical
  700 MB, because a batch's scan dir now lives in the job's scratch. Per-job logs
  land in `--htcondor-jobdir` (the entrypoints point that at `<log_dir>/outfiles`).
  String ClassAd values are passed **bare**; the plugin quotes them itself.
  Note that lxplus overrides two of the requests: `MaxRuntime` is derived from
  the JobFlavour (espresso shows 1200, not the profile's 604800) and
  `RequestMemory` comes back higher than `htcondor_request_mem_mb`.

## Orchestrator lifecycle (cluster backends)

The historical `condor_submit` was fire-and-forget; snakemake needs a driver
process to submit/monitor/publish. Cluster submissions therefore **detach** it
by default (`setsid`, log at `<log_dir>/snakemake.log` — the path is printed at
submission; `--foreground` to keep it attached). If the driver dies (e.g.
submit-node reboot), the jobs in flight finish on the cluster; nothing new gets
scheduled until you resume — by **re-running the exact same command** (finished
batches are detected via their outputs/manifests and skipped; add
`snakemake --unlock` if a stale lock is reported).

Each cluster job runs its batch's *whole* rule group (see the profiles' `groups:`),
so already-submitted batches complete end to end — including `tarball` and
`publish` — with no driver present. What a dead driver costs you is the
scheduling of anything not yet submitted, the `retries`, and the final target
check.

### Checking a run that lost its driver

`retries: 2` is an orchestrator-side setting, and the remote jobs are launched
with `--retries 0`, so **a batch that fails while no driver is alive is not
retried and nothing reports it** — you get N-1 of N with no error anywhere. Only
infrastructure errors exit non-zero (per-model tool failures are data, exit 0),
so this is rare, but it is silent. Count the products rather than assume:

```bash
cd <scan_parent>
ls -d scan.<runid>.*/ | wc -l                  # batches that got as far as sampling
ls scan.<runid>.*/ntuple.<runid>.*.root | wc -l   # ...that finished (no --publish)
ls r3mg/scan.<runid>.*/published.ok | wc -l       # ...that published (--publish)
```

Anything short of `--num_jobs` means a batch is missing. The remedy is always the
same and is safe to run at any time: **re-run the exact same `genModels.py`
command**. Finished batches are skipped via their manifests, so it only rebuilds
what is absent. Each batch's `r3mg/scan.<runid>.<b>/step_*.json` records the
scheduler `job_id` that ran it, so a failed batch maps straight to its
`condor_q`/`sacct -j` history and its log under `r3mg/logs/<runid>/outfiles/`.

> **Enable lingering if you intend to log out.** Detaching uses
> `start_new_session`, which survives SIGHUP, but systemd-logind reaps a user's
> processes when their last session on the node ends unless the account lingers.
> Check with `loginctl show-user $USER | grep Linger`; enable once with
> `loginctl enable-linger $USER`. Without it, logging off the submit node kills
> the orchestrator — the submitted jobs still finish, but the run stops
> progressing until you re-run the command. (`--foreground` under `tmux`/`screen`
> is the other way to keep a long run driven.)

Batch runs are guarded by `<scan_parent>/r3mg/run.<runid>.yaml`: re-submitting
the same runid with a different config content or `--base_seed` is refused (it
would re-sample and wipe finished batches — use a new runid); growing
`--num_jobs` tops the run up with new batches; adding `--publish` later
publishes an already-generated run. A **foreground** driver killed with Ctrl-C
actively cancels the cluster jobs (deliberate abort). One run at a time per
invocation directory (snakemake locks the cwd).

