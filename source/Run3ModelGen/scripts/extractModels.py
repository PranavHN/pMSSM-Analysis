#!/usr/bin/env python

import base64
import os
import sys

import click
from click.core import ParameterSource

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)


def _explicit(ctx, name):
    return ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE


@click.command()
@click.option('--scan_dir', type=str, default=None, help='Scan directory to extract models from. Can also be a .tar ball.')
@click.option('--root_file', type=str, default=None, help='Root NTuple with scan info')
@click.option('--selection', type=str, default="akarr['SS_m_h']!=-1", help='Selection to apply.')
@click.option('--eos_dir', type=str, default=None, help='Batch mode: EOS run directory containing per-ProcId subdirs (e.g. $EOSPATH/<runid>); one extraction job per subdir.')
@click.option('--backend', default="local", type=click.Choice(("local", "slurm", "htcondor")), help='Batch mode: where the extraction jobs run.')
@click.option('--num_jobs', default=None, type=click.IntRange(min=1), help='Batch mode: number of ProcIds (0..N-1). Default: all digit subdirs of --eos_dir.')
@click.option('--flavour', default="tomorrow", type=str, help='HTCondor JobFlavour (htcondor backend only).')
@click.option('--log_dir', default=None, type=str, help='Batch mode: orchestrator/scheduler logs + extraction markers (default: extractModels_logs/<ClusterId>).')
@click.option('--foreground', is_flag=True, help='Batch mode, cluster backends: keep the orchestrator attached instead of detaching it.')
@click.pass_context
def main(ctx, scan_dir, root_file, selection, eos_dir, backend, num_jobs, flavour, log_dir, foreground):
    '''Main script for extracting interesting models from a scan.

    Two modes, one entrypoint:
      * default: extract from ONE scan dir/tarball + ntuple, in-process
        (exactly the historical behavior — no snakemake involved);
      * --eos_dir: fan out one extraction job per ProcId subdirectory of a
        published run, on --backend local|slurm|htcondor. (Those jobs invoke
        this script's single mode — no nested snakemake.)
    '''

    if eos_dir is None:
        # ── single mode: unchanged in-process extraction ───────────────────
        for flag in ("num_jobs", "foreground", "log_dir", "flavour"):
            if _explicit(ctx, flag):
                raise click.UsageError(f"--{flag} applies to batch extraction; add --eos_dir <run dir>.")
        if _explicit(ctx, 'backend') and backend != "local":
            raise click.UsageError("single-file extraction runs in-process; batch extraction over a run needs --eos_dir.")
        if scan_dir is None or root_file is None:
            raise click.UsageError("either extract one scan (--scan_dir + --root_file) or a published run (--eos_dir).")

        from Run3ModelGen.extractor import ModelExtractor

        # Initialise ModelExtractor
        ModExt = ModelExtractor(scan_dir=scan_dir, root_file=root_file, selection=selection)

        # Select interesting models
        ModExt.select_models()

        # Move interesting models to EOS
        ModExt.move_models()
        return

    # ── batch mode: one job per ProcId subdir of eos_dir ───────────────────
    from Run3ModelGen import snakefacade

    datadir = os.environ['DATAPATH'].split(':')[0]

    if scan_dir is not None or root_file is not None:
        raise click.UsageError("--eos_dir extracts a whole run; --scan_dir/--root_file select the single-scan mode — pass one or the other.")

    eos_dir = eos_dir if not eos_dir.endswith('/') else eos_dir[:-1]
    if eos_dir.startswith("root://") or not os.path.isdir(eos_dir):
        raise click.UsageError(f"--eos_dir must be a directory reachable from this host, got '{eos_dir}'.")
    ClusterId = eos_dir.split('/')[-1]

    available = sorted((d for d in os.listdir(eos_dir)
                        if os.path.isdir(f"{eos_dir}/{d}") and d.isdigit()), key=int)
    if num_jobs is None:
        procids = available
    else:
        procids = [str(p) for p in range(num_jobs)]
        missing = [p for p in procids if p not in available]
        if missing:
            raise click.UsageError(f"--num_jobs {num_jobs} expects ProcId dirs 0..{num_jobs-1} under {eos_dir}, "
                                   f"but these are missing: {', '.join(missing)}.")
    if not procids:
        raise click.ClickException(f"No ProcId subdirectories found in {eos_dir}!")

    problems = snakefacade.preflight(backend)
    if problems:
        for p in problems:
            log.fatal(p)
        raise click.ClickException("pre-flight checks failed (nothing was submitted).")

    # Print logo. Note: ASCII art generated with https://patorjk.com/software/taag/ (Small, Fitted)
    with open(f"{datadir}/logo.txt", 'r') as file:
        print("\n" + file.read() + "\n")

    log_dir = os.path.abspath(log_dir or f"extractModels_logs/{ClusterId}")
    os.makedirs(f"{log_dir}/outfiles", exist_ok=True)

    # base64 so the quotes in the awkward expression never reach a scheduler
    # submit description (HTCondor rejects them); extract.smk decodes it.
    cfg = {"mode": "extract", "eos_dir": os.path.abspath(eos_dir),
           "selection_b64": base64.b64encode(selection.encode()).decode(),
           "procids": ",".join(procids), "work_dir": log_dir}
    argv = snakefacade.snakemake_argv(backend, cfg, targets=["extract_all"],
                                      extra=snakefacade.scheduler_extra(backend, log_dir, flavour))

    log.info(f"Submitting {len(procids)} extraction jobs for cluster {ClusterId} via snakemake [backend: {backend}]...")

    if backend == "local" or foreground:
        if backend != "local":
            log.warning(f"Foreground orchestrator: Ctrl-C cancels the submitted {backend} jobs.")
        rc = snakefacade.run_foreground(argv)
        if rc == 0:
            log.info(f"The Models directory to be copied into the pMSSMFactory is saved in {os.environ['EOSPATH']}/SelectedModels/{ClusterId}/Models")
        sys.exit(rc)

    pid = snakefacade.run_detached(argv, f"{log_dir}/snakemake.log", warn=log.warning)
    log.info(f"Snakemake orchestrator detached (pid {pid}).")
    log.info(f"Progress: tail -f {log_dir}/snakemake.log (per-job logs in {log_dir}/outfiles/; "
             f"queue: {'squeue --me' if backend == 'slurm' else 'condor_q'})")
    log.info(f"To resume after an interruption: re-run this exact command (finished ProcIds are skipped).")
    log.info(f"When finished, the Models directory to be copied into the pMSSMFactory will be saved in {os.environ['EOSPATH']}/SelectedModels/{ClusterId}/Models")

if __name__ == "__main__":
    main()
