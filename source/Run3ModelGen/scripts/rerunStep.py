#!/usr/bin/env python

import os
import sys

import click
import yaml

from Run3ModelGen import pipeline
from Run3ModelGen import snakefacade

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)

datadir = os.environ['DATAPATH'].split(':')[0]

@click.command()
@click.option('--step', 'step_name', required=True, type=str, help="Pipeline step to re-run (e.g. micromegas), or 'ntuple' to only rebuild the ntuple.")
@click.option('--scan_dir', default="scan", type=str, help='Scan directory of the existing scan.')
@click.option('--config_file', default=None, help='Scan config yaml of the original run. Default: recovered from <scan_dir>/scan_config.yaml.')
@click.option('--seed', default=None, type=int, help='Seed of the original run. Default: recovered from <scan_dir>/scan_config.yaml.')
@click.option('--backend', default="local", type=click.Choice(snakefacade.BACKENDS), help='Where the re-run executes; cluster backends submit the step tail as one grouped job.')
@click.option('--log_dir', default=None, type=str, help='Cluster backends: orchestrator/scheduler logs (default: under the scan bookkeeping).')
@click.option('--foreground', is_flag=True, help='Cluster backends: keep the snakemake orchestrator attached instead of detaching it.')
def main(step_name, scan_dir, config_file, seed, backend, log_dir, foreground):
    '''Re-run ONE pipeline step of an existing scan — plus everything after it
    in the configured step order (nothing before it) — then rebuild the ntuple.

    Counterpart to genModels.py, which always regenerates the whole scan from
    scratch. The original run's config file and seed are recovered from the
    scan's scan_config.yaml, so a plain
    `rerunStep.py --step micromegas --scan_dir scan` is usually enough.

    Not applicable to prior=mcmc scans: the chain runs the whole pipeline
    inside sampling and can only be redone as a whole (via genModels.py).
    '''

    # Absolute for the same reason as in genModels.py: a cluster worker runs in
    # its own scratch dir, so a relative scan_dir would send the re-run's outputs
    # there instead of into the existing scan.
    scan_dir = os.path.abspath(scan_dir)

    problems = snakefacade.preflight(backend)
    if problems:
        for p in problems:
            log.fatal(p)
        raise click.ClickException("pre-flight checks failed (nothing was modified).")

    # Don't delete manifests under a run that is currently executing from here.
    if snakefacade.locks_active("."):
        log.fatal("another snakemake run appears active from this directory (.snakemake/locks is "
                  "non-empty) — wait for it, or 'snakemake --unlock' if it is a stale lock.")
        exit(1)
    name = os.path.basename(scan_dir)
    parent = os.path.dirname(scan_dir) or "."
    meta = os.path.normpath(f"{parent}/r3mg/{name}")

    # Recover the original run parameters from the scan itself. Getting these
    # wrong would outdate the sample rule, which wipes and regenerates the
    # whole scan — exactly what this script exists to avoid.
    recorded_file = f"{scan_dir}/scan_config.yaml"
    if not os.path.isfile(recorded_file):
        log.fatal(f"{recorded_file} not found — is '{scan_dir}' a scan directory produced by the workflow?")
        exit(1)
    with open(recorded_file) as fh:
        recorded = yaml.safe_load(fh)

    if recorded.get('prior') == 'mcmc':
        log.fatal("prior=mcmc scans run the pipeline inside the sampling chain and have no per-step "
                  "re-run; regenerate the whole chain with genModels.py instead.")
        # The one thing people usually come here for is redoing the diagnostics. They are
        # the tail of the `chain` rule rather than a step, and `diag/` is a declared rule
        # output (invalidating it re-runs the sampler), so point at the direct command.
        log.info(f"To redo only the emcee diagnostics of this scan (seconds; reads chain.h5 "
                 f"read-only, re-runs no sampling):  pixi run rediagnose {scan_dir}")
        exit(1)

    if seed is None:
        seed = recorded.get('seed')
        if seed is None:
            log.fatal(f"no seed recorded in {recorded_file}; pass --seed explicitly.")
            exit(1)
    if config_file is None:
        config_file = recorded.get('config_file')
    if config_file is not None and not os.path.isfile(config_file):
        log.fatal(f"config file '{config_file}' (recorded in {recorded_file}) not found; "
                  f"pass --config_file explicitly.")
        exit(1)

    # Validate the step against this scan's (platform-resolved) pipeline.
    cfg = pipeline.load_scan_config(config_file, f"{datadir}/configs/default_config.yaml")
    step_names = [s['name'] for s in pipeline.resolve_steps(cfg.get('steps', []))]

    # Snakemake does not re-validate finished jobs against missing inputs, so
    # removing ONLY the target step's manifest would leave it unscheduled (its
    # downstream consumers still have their outputs). Removing the manifests of
    # the step AND everything after it — plus the ntuple — forces exactly the
    # tail of the chain to re-run, deterministically.
    ntuple = pipeline.ntuple_name(scan_dir)
    if step_name == 'ntuple':
        if os.path.exists(ntuple):
            os.remove(ntuple)
        log.info(f"Will rebuild: ntuple only")
    elif step_name in step_names:
        # Everything BEFORE the target step must be intact — that's what the
        # re-run builds on. (The target's own manifest may already be absent,
        # e.g. from an interrupted run; it gets regenerated either way.)
        idx = step_names.index(step_name)
        if not os.path.exists(f"{meta}/points.npz"):
            log.fatal(f"no {meta}/points.npz — this scan was not produced by the workflow; regenerate it with genModels.py.")
            exit(1)
        if idx > 0 and not os.path.exists(f"{meta}/step_{step_names[idx-1]}.json"):
            log.fatal(f"the preceding step '{step_names[idx-1]}' has no manifest — re-run from that step "
                      f"(or an earlier one) instead, or regenerate the scan with genModels.py.")
            exit(1)
        downstream = step_names[idx:]
        for s in downstream:
            m = f"{meta}/step_{s}.json"
            if os.path.exists(m):
                os.remove(m)
        if os.path.exists(ntuple):
            os.remove(ntuple)
        log.info(f"Will re-run: {' -> '.join(downstream)} -> ntuple")
    else:
        log.fatal(f"step '{step_name}' is not in this scan's pipeline {step_names} (or 'ntuple').")
        exit(1)

    cli_cfg = {"mode": "single", "scan_dir": scan_dir, "seed": seed}
    if config_file is not None:
        cli_cfg["config_file"] = os.path.abspath(config_file)

    # --rerun-triggers mtime: with the default triggers (params/code/software-env)
    # any code or config drift since the original run would outdate the SAMPLE
    # rule — which wipes the scan dir, the exact thing this script must never do.
    # mtime-only makes the re-run purely "produce the deleted outputs".
    if backend == "local":
        argv = snakefacade.snakemake_argv(backend, cli_cfg, extra=["--rerun-triggers", "mtime"])
        sys.exit(snakefacade.run_foreground(argv))

    log_dir = os.path.abspath(log_dir or os.path.normpath(f"{meta}/logs"))
    os.makedirs(f"{log_dir}/outfiles", exist_ok=True)
    argv = snakefacade.snakemake_argv(backend, cli_cfg,
                                      extra=["--rerun-triggers", "mtime"] + snakefacade.scheduler_extra(backend, log_dir))
    log.info(f"Submitting the re-run as one {backend} job via snakemake...")
    if foreground:
        log.warning(f"Foreground orchestrator: Ctrl-C cancels the submitted {backend} job.")
        sys.exit(snakefacade.run_foreground(argv))
    pid = snakefacade.run_detached(argv, f"{log_dir}/snakemake.log", warn=log.warning)
    log.info(f"Snakemake orchestrator detached (pid {pid}).")
    log.info(f"Progress: tail -f {log_dir}/snakemake.log (per-job logs in {log_dir}/outfiles/; "
             f"queue: {'squeue --me' if backend == 'slurm' else 'condor_q'})")

if __name__ == "__main__":
    main()
