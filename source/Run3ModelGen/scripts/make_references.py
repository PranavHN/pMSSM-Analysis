#!/usr/bin/env python

'''Generate reference outputs for all CI scans and print per-step success counts.

For each CI scan setting this runs genModels (exactly as CI does) and prints the
per-step success counts, leaving the reference ntuples under run/scan_<name>/. It
does NOT upload anything: review the printed summary, then run `pixi run upload-refs`
to upload the references to the shared nicebot EOS area.

Manual task: `pixi run make-refs`.
'''

import os
import shutil
import subprocess

import click

from Run3ModelGen.refcheck import (
    SCANS, SEED_CSV, by_name, per_step_counts, format_counts_table,
)

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)


def _run_scan(scan, run_dir):
    '''Run genModels for one scan inside run_dir, matching the CI invocation.'''
    if scan["needs_seed_csv"]:
        src = os.path.normpath(os.path.join(run_dir, SEED_CSV))
        shutil.copy(src, run_dir)
        log.info(f"copied {src} -> {run_dir}/")

    argv = ["genModels.py", "--scan_dir", scan["scan_dir"]]
    if scan["config"]:
        argv += ["--config_file", scan["config"]]
    log.info(f"[{scan['name']}] running: {' '.join(argv)} (cwd={run_dir})")
    subprocess.run(argv, cwd=run_dir, check=True)


@click.command()
@click.option('--scans', default='all', help="Comma-separated scan names to (re)make, or 'all'.")
@click.option('--run_dir', default='run', help='Working directory for the scans (default: run).')
def main(scans, run_dir):
    '''Produce reference scans and print per-step success counts (no upload).'''
    selected = SCANS if scans == 'all' else [by_name(n.strip()) for n in scans.split(',')]
    os.makedirs(run_dir, exist_ok=True)

    summary = []
    for scan in selected:
        name = scan["name"]
        log.info(f"=== reference scan '{name}' -> {scan['scan_dir']} ===")
        _run_scan(scan, run_dir)
        counts = per_step_counts(os.path.join(run_dir, scan["scan_dir"]))
        print(format_counts_table(counts, title=f"[{name}] per-step success counts (passed/attempted):"))
        summary.append((name, counts))

    print("\n=== reference summary ===")
    for name, counts in summary:
        print(format_counts_table(counts, title=f"[{name}]:"))

    scans_arg = "" if scans == 'all' else f" --scans {scans}"
    print("\nReview the summary above. When happy, upload these references with:")
    print("    export CERN_USER=nicebot")
    print("    export SERVICE_PASS=\"$(printf %s 'THE_NICEBOT_PASSWORD' | base64 -w0)\"")
    print(f"    pixi run upload-refs{scans_arg}")


if __name__ == "__main__":
    main()
