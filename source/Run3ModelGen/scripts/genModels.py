#!/usr/bin/env python

import os
import shlex
import shutil
import sys

import click
import yaml
from click.core import ParameterSource

from Run3ModelGen import pipeline
from Run3ModelGen import snakefacade

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)

datadir = os.environ['DATAPATH'].split(':')[0]


def _explicit(ctx, name):
    return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE


# Default batch-data root for --backend htcondor. HTCondor exports
# _CONDOR_SCRATCH_DIR per job and deletes the directory when the job exits, so this
# must reach snakemake UNEXPANDED: the workflow expandvars-resolves it when each
# worker re-parses the Snakefile, giving every batch its own node-local scan dir.
# Nothing on the submit host ever resolves or touches it.
CONDOR_SCRATCH_SCAN = "$_CONDOR_SCRATCH_DIR/scan"

# Rough per-batch footprint of a scan dir at the 10k-model cap, for size warnings.
_GB_PER_BATCH = 1


@click.command()
@click.option('--config_file', default=None, help='Yaml file containing scan configuration. If no config is provided, will read from the default.')
@click.option('--scan_dir', default="scan", type=str, help='Scan output directory (with --num_jobs > 1: the parent directory holding the scan.<runid>.<b> batch dirs). Batch runs on --backend htcondor default instead to each worker\'s $_CONDOR_SCRATCH_DIR (requires --publish); naming a shared-storage directory here opts back out, and warns about the size.')
@click.option('--seed', default=123, type=int, help='Seed for np random number generation (single-scan mode only).')
@click.option('--backend', default="local", type=click.Choice(snakefacade.BACKENDS), help='Where to run: locally, or submitted via SLURM/HTCondor.')
@click.option('--num_jobs', default=1, type=click.IntRange(min=1), help='Number of batch jobs; each generates num_models models from the config.')
@click.option('--runid', default=None, type=str, help='Digits-only run id for batch runs (default: timestamp). (runid, base_seed, config) reproducibly determine every sampled point.')
@click.option('--base_seed', default=123, type=int, help='Batch runs: per-batch seed = blake2b(runid:base_seed:batch).')
@click.option('--publish', is_flag=True, help='Batch runs: move tarball + ntuple + config to $EOSPATH/<runid>/<b>/ (any backend); each batch\'s local scan dir and tarball are removed once its published copies verify.')
@click.option('--keep_local', is_flag=True, help='With --publish: keep every batch\'s scan dir and tarball on local/shared disk as well as on EOS (needs room for the whole run: ~1 GB per 10k-model batch).')
@click.option('--flavour', default="tomorrow", type=str, help='HTCondor JobFlavour (htcondor backend only). Use "espresso" for testing.')
@click.option('--log_dir', default=None, type=str, help='Cluster backends: orchestrator + per-job scheduler logs (default: under <scan parent>/r3mg/).')
@click.option('--foreground', is_flag=True, help='Cluster backends: keep the snakemake orchestrator attached instead of detaching it (Ctrl-C then cancels the jobs).')
@click.pass_context
def main(ctx, config_file, scan_dir, seed, backend, num_jobs, runid, base_seed, publish, keep_local, flavour, log_dir, foreground):
    '''Main script for generating models according to provided config file. Will also generate root NTuple.

    One entrypoint for every backend: --backend local|slurm|htcondor selects
    the executor, behavior is otherwise identical. --num_jobs 1 (default)
    generates one scan at --scan_dir, overwriting it (fresh-scan semantics);
    --num_jobs N > 1 fans out N batches under --scan_dir without ever wiping —
    re-running the same command resumes an interrupted run.

    Batch runs on --backend htcondor write each batch to its own worker's
    $_CONDOR_SCRATCH_DIR rather than to --scan_dir, so N concurrent batches cost
    the submit volume nothing but bookkeeping; only the published copies leave the
    node, which is why that default requires --publish. Pass --scan_dir explicitly
    to keep the batch data on shared storage instead.
    '''
    batch = num_jobs > 1

    # HTCondor batch runs put the batch data on the worker by DEFAULT: a scan dir is
    # ~1 GB per 10k-model batch, and N of them landing on shared storage at once is
    # what makes a wide run run the submit volume out of quota. Only the published
    # copies then leave the node, so this requires --publish (checked below).
    # An explicit --scan_dir opts back into the shared-FS layout, with a warning.
    scratch_scan = backend == "htcondor" and batch and not _explicit(ctx, 'scan_dir')

    if not scratch_scan:
        # ABSOLUTE, always (batch mode does the same for its parent below). A cluster
        # worker does not run in the submit directory — HTCondor gives the job its own
        # scratch dir — while snakemake is told the filesystem is shared and stages
        # nothing. A relative scan_dir therefore resolves against the worker's scratch:
        # the scan, and the r3mg bookkeeping that makes the run resumable, would be
        # built somewhere the orchestrator cannot see. (scratch_scan asks for exactly
        # that placement, deliberately, and keeps the bookkeeping behind on shared
        # storage instead — see the batch-mode section.)
        scan_dir = os.path.abspath(scan_dir)

    # ── flag-combination validation (before ANY side effect) ──────────────
    if _explicit(ctx, 'seed') and batch:
        raise click.UsageError("--seed applies to single-scan mode; batch runs derive per-batch seeds — use --base_seed.")
    if _explicit(ctx, 'base_seed') and not batch:
        raise click.UsageError("--base_seed applies to batch runs (--num_jobs > 1); single-scan mode uses --seed.")
    if _explicit(ctx, 'runid') and not batch:
        raise click.UsageError("--runid names batch runs (--num_jobs > 1); single-scan mode has no run id.")
    if publish and not batch:
        raise click.UsageError(
            "--publish uses the batch EOS layout ($EOSPATH/<runid>/<b>/) and needs --num_jobs >= 2. "
            "For a single published batch, run the workflow directly: pixi run smk --workflow-profile "
            "../workflow/profiles/<backend> --config mode=batch batches=1 runid=<id> publish=eos "
            "scan_parent=<dir> config_file=...")
    if keep_local and not publish:
        raise click.UsageError(
            "--keep_local only means something with --publish: without it nothing is "
            "copied to EOS and the scan dirs are always kept.")
    if scratch_scan and not publish:
        raise click.UsageError(
            f"--backend htcondor writes each batch into its worker's $_CONDOR_SCRATCH_DIR, "
            f"which HTCondor destroys when the job exits — without --publish the run would "
            f"produce nothing you can read afterwards. Either add --publish (each batch's "
            f"tarball + ntuple go to $EOSPATH/<runid>/<b>/), or name a shared-storage "
            f"--scan_dir to keep the batches on disk instead (~{_GB_PER_BATCH} GB per "
            f"10k-model batch, so ~{_GB_PER_BATCH * num_jobs} GB for this run).")
    if scratch_scan and keep_local:
        # Unreachable via the check above (keep_local implies publish), but explicit:
        # keeping a copy of a directory that dies with the job is meaningless, and it
        # would also put the ntuples back in the target list at paths that only ever
        # exist on a worker — the orchestrator would wait for them forever.
        raise click.UsageError(
            "--keep_local keeps every batch's scan dir and tarball next to the run, but the "
            "htcondor default writes them to per-job scratch that dies with the job. Name a "
            "shared-storage --scan_dir to use --keep_local.")
    if _explicit(ctx, 'flavour') and backend != "htcondor":
        raise click.UsageError(f"--flavour sets the HTCondor JobFlavour; the {backend} backend takes its "
                               f"runtimes from workflow/profiles/{backend}/config.yaml.")
    if _explicit(ctx, 'log_dir') and backend == "local":
        log.warning("--log_dir is ignored on the local backend (output goes to the terminal).")
    if backend == "htcondor" and batch and not scratch_scan:
        # Opting out of per-job scratch onto shared storage. This is the combination
        # that fills volumes: every batch keeps its whole scan dir there at once.
        freed = (" They are freed per batch as --publish verifies its EOS copies, but the "
                 "peak is what has to fit." if publish else
                 " Nothing is freed during the run, and without --publish nothing is copied "
                 "off it either.")
        log.warning(
            f"--scan_dir puts the batch data on shared storage: all {num_jobs} batches write "
            f"their scan dirs under {scan_dir} concurrently. That is routinely much larger "
            f"than expected — about {_GB_PER_BATCH} GB per 10k-model batch, so on the order of "
            f"{_GB_PER_BATCH * num_jobs} GB at peak for this run.{freed} Omit --scan_dir to "
            f"write each batch to its own worker's $_CONDOR_SCRATCH_DIR instead.")
    if runid is not None and not runid.isdigit():
        raise click.UsageError(f"--runid must be digits only (ClusterId-shaped), got '{runid}'.")
    if config_file is not None and not os.path.isfile(config_file):
        raise click.UsageError(f"config file '{config_file}' does not exist or is not a file!")

    # ── pre-flight (still before any side effect) ──────────────────────────
    problems = snakefacade.preflight(backend, publish=publish)
    default_config = f"{datadir}/configs/default_config.yaml"
    merged = pipeline.load_scan_config(config_file, default_config)
    if merged.get('num_models', 0) > pipeline.MAX_MODELS_PER_JOB:
        problems.append(f"num_models={merged['num_models']} exceeds the {pipeline.MAX_MODELS_PER_JOB} "
                        f"per-batch cap — reduce it and scale out with --num_jobs instead.")
    if problems:
        for p in problems:
            log.fatal(p)
        raise click.ClickException("pre-flight checks failed (nothing was modified).")

    cluster = backend != "local"
    if cluster or batch:
        # Print logo. Note: ASCII art generated with https://patorjk.com/software/taag/ (Small, Fitted)
        with open(f"{datadir}/logo.txt", 'r') as file:
            print("\n" + file.read() + "\n")

    if not batch:
        # ── single-scan mode: fresh-scan overwrite semantics, as always ────
        parent = os.path.dirname(scan_dir) or "."
        name = os.path.basename(scan_dir)
        pipeline.wipe_scan_dir(scan_dir)          # refuses to destroy an emcee chain.h5
        shutil.rmtree(f"{parent}/r3mg/{name}", ignore_errors=True)

        cfg = {"mode": "single", "scan_dir": scan_dir, "seed": seed}
        if config_file is not None:
            cfg["config_file"] = os.path.abspath(config_file)

        if not cluster:
            sys.exit(snakefacade.run_foreground(snakefacade.snakemake_argv("local", cfg)))

        log_dir = os.path.abspath(log_dir or os.path.normpath(f"{parent}/r3mg/{name}/logs"))
        os.makedirs(f"{log_dir}/outfiles", exist_ok=True)
        argv = snakefacade.snakemake_argv(backend, cfg,
                                          extra=snakefacade.scheduler_extra(backend, log_dir, flavour))
        log.info(f"Submitting scan {scan_dir} as one {backend} job via snakemake...")
        log.info(f"Note: re-running genModels.py restarts this scan from scratch; to RESUME an "
                 f"interrupted scan run instead: {shlex.join(argv)}")
        if foreground:
            log.warning(f"Foreground orchestrator: Ctrl-C cancels the submitted {backend} jobs.")
            sys.exit(snakefacade.run_foreground(argv))
        pid = snakefacade.run_detached(argv, f"{log_dir}/snakemake.log", warn=log.warning)
        log.info(f"Snakemake orchestrator detached (pid {pid}).")
        log.info(f"Progress: tail -f {log_dir}/snakemake.log (per-job logs in {log_dir}/outfiles/; "
                 f"queue: {'squeue --me' if backend == 'slurm' else 'condor_q'})")
        log.info(f"When finished: {scan_dir}/{os.path.basename(pipeline.ntuple_name(scan_dir))}")
        return

    # ── batch mode: N batches under <scan_dir>/, resume-friendly ──────────
    # Two roots with different lifetimes (see the Snakefile docstring). `parent` is
    # the PERSISTENT one — run metadata, logs, published markers, resume state — and
    # everything submit-side keys off it. `scan_parent` holds the batch data and may
    # be per-job and ephemeral. They are the same directory unless scratch_scan.
    if scratch_scan:
        # Bookkeeping stays exactly where a shared-FS run would have put it (the
        # --scan_dir default, i.e. ./scan), so logs, run metadata and resume are
        # unchanged; only the bulky batch dirs move onto the workers.
        scan_parent = CONDOR_SCRATCH_SCAN
        parent = os.path.abspath(scan_dir)
    else:
        scan_parent = parent = os.path.abspath(scan_dir)
    os.makedirs(parent, exist_ok=True)
    meta_dir = f"{parent}/r3mg"
    runid = runid or snakefacade.default_runid()
    log_dir = os.path.abspath(log_dir or f"{meta_dir}/logs/{runid}")
    os.makedirs(f"{log_dir}/outfiles", exist_ok=True)

    current = {
        "runid": runid, "base_seed": base_seed, "num_jobs": num_jobs,
        "publish": bool(publish), "keep_local": bool(keep_local), "backend": backend,
        "config_sha256": snakefacade.config_digest(config_file, default_config),
        "created": snakefacade.default_runid(),
    }

    meta_path = snakefacade.run_meta_path(meta_dir, runid)
    existing = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            existing = yaml.safe_load(fh)
        fatal, warnings = snakefacade.check_run_meta(existing, current)
        for w in warnings:
            log.warning(w)
        if fatal:
            for f in fatal:
                log.fatal(f)
            raise click.ClickException(f"run {runid} already exists in {parent} and cannot be safely re-submitted.")
        log.info(f"Resuming run {runid} in {parent} (finished batches are skipped).")

    # Snapshot the config next to the logs; jobs read the snapshot, so later
    # edits of the original cannot change a run in flight. Reuse the existing
    # snapshot on resume (the content hash was verified above).
    if config_file is not None:
        run_config = (existing or {}).get("config_copy")
        if not (run_config and os.path.isfile(run_config)):
            run_config = f"{log_dir}/{os.path.basename(config_file)}"
            shutil.copy(config_file, run_config)
        current["config_copy"] = run_config
        run_config = os.path.abspath(run_config)
    else:
        run_config = None

    # Publishing into an existing run directory overwrites its files (cp/xrdcp -f).
    # Harmless when resuming this very run, suspicious otherwise (e.g. a reused
    # runid from a different parent, where the run-meta guard cannot see it).
    if publish and existing is None:
        eospath = os.environ['EOSPATH'].rstrip('/')
        if not eospath.startswith("root://") and os.path.isdir(f"{eospath}/{runid}"):
            log.warning(f"{eospath}/{runid} already exists and is not from a run recorded in {parent} — "
                        f"publishing will overwrite files in it. Use a different --runid if that is unintended.")

    # Heads-up when other runs share this parent (fully namespaced, so allowed).
    others = sorted({d.split(".")[1] for d in os.listdir(parent)
                     if d.startswith("scan.") and len(d.split(".")) == 3 and d.split(".")[1] != runid})
    if others:
        log.info(f"Note: {parent} also holds batches of run(s) {', '.join(others)} — fine, all outputs are namespaced by runid.")

    cfg = {"mode": "batch", "runid": runid, "batches": num_jobs, "base_seed": base_seed,
           "scan_parent": scan_parent, "meta_parent": parent,
           "publish": "eos" if publish else "none",
           "keep_local": "1" if keep_local else "0", "config_file": run_config}
    argv = snakefacade.snakemake_argv(backend, cfg,
                                      extra=snakefacade.scheduler_extra(backend, log_dir, flavour))

    substr = f"0-{num_jobs-1}" if num_jobs > 1 else "0"
    flavour_note = f", JobFlavour {flavour}" if backend == "htcondor" else ""
    log.info(f"Submitting run {runid} ({num_jobs} batch jobs{flavour_note}) via snakemake [backend: {backend}]...")
    if scratch_scan:
        log.info(f"Each batch runs entirely in its own worker's $_CONDOR_SCRATCH_DIR — nothing but "
                 f"its published copies leaves the node, so this run needs no room under {parent} "
                 f"beyond the bookkeeping in {meta_dir}. An evicted batch leaves nothing behind "
                 f"and simply re-runs whole on resume.")

    if backend == "local" or foreground:
        if cluster:
            log.warning(f"Foreground orchestrator: Ctrl-C cancels the submitted {backend} jobs.")
        current["orchestrator_pid"] = None
        snakefacade.write_run_meta(meta_dir, runid, current)
        rc = snakefacade.run_foreground(argv)
        if rc == 0 and publish:
            log.info(f"The .tar ball and ntuple of every batch are saved in {os.environ['EOSPATH']}/{runid}/{substr}")
            log.info("Each batch's scan dir and tarball lived on its worker's scratch and went "
                     "with the job; the published copies are the run's only output."
                     if scratch_scan else
                     f"Local scan dirs and tarballs were removed after their published copies verified."
                     if not keep_local else
                     f"--keep_local: the scan dirs and tarballs are also still in {parent}.")
        sys.exit(rc)

    pid = snakefacade.run_detached(argv, f"{log_dir}/snakemake.log", warn=log.warning)
    current["orchestrator_pid"] = pid
    snakefacade.write_run_meta(meta_dir, runid, current)
    log.info(f"Snakemake orchestrator detached (pid {pid}); it submits, monitors"
             f"{' and publishes' if publish else ''} the jobs.")
    log.info(f"Progress: tail -f {log_dir}/snakemake.log (per-job logs in {log_dir}/outfiles/; "
             f"queue: {'squeue --me' if backend == 'slurm' else 'condor_q'})")
    log.info(f"To resume after an interruption: re-run this exact command (finished batches are skipped).")
    if publish:
        log.info(f"After your models have finished generating, the .tar ball and ntuple will be saved in {os.environ['EOSPATH']}/{runid}/{substr}")
        log.info("Each batch's scan dir and tarball stay on its worker and go with the job; the "
                 "published copies are the run's only output."
                 if scratch_scan else
                 "Each batch's local scan dir and tarball are removed as soon as its published "
                 "copies verify (--keep_local to keep them too)."
                 if not keep_local else
                 f"--keep_local: the scan dirs and tarballs are kept in {parent} as well.")
    else:
        log.info(f"When finished: {parent}/scan.{runid}.<b>/ntuple.{runid}.<b>.root for b = {substr}")

if __name__ == "__main__":
    main()
