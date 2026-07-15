#!/usr/bin/env python

'''Compare a freshly produced scan against its reference on EOS (per-step counts).

Used by the CI genModels jobs: downloads the reference per-step counts for the
given scan from the shared nicebot EOS area, recomputes the per-step success
counts of the just-produced scan, prints both, and exits non-zero on any
mismatch (which fails the CI job).

Requires a valid Kerberos ticket (the CI job does the kinit) and xrdcp on PATH.
'''

import os
import sys
import json
import tempfile
import subprocess

import click

from Run3ModelGen.refcheck import (
    by_name, counts_url, per_step_counts, compare_counts, format_counts_table,
    counts_from_json,
)

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)


@click.command()
@click.option('--scan', 'scan_name', required=True,
              help='Scan name: flat | fixed | mcmc | grid | importance.')
@click.option('--run_dir', default='.',
              help="Directory containing the scan_<name>/ dir (default: current dir; the CI jobs run this from inside run/).")
def main(scan_name, run_dir):
    '''Compare the produced scan's per-step counts against the EOS reference.'''
    scan = by_name(scan_name)
    scan_dir = os.path.join(run_dir, scan["scan_dir"])

    # Per-step counts of the just-produced scan.
    produced = per_step_counts(scan_dir)

    # Reference per-step counts (download the JSON sidecar from EOS).
    url = counts_url(scan_name)
    with tempfile.TemporaryDirectory() as tmp:
        dest = os.path.join(tmp, "counts.json")
        try:
            subprocess.run(["xrdcp", "-f", url, dest], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log.error(
                f"[{scan_name}] could not fetch reference from {url}: {exc}. "
                "Has it been seeded? Run `pixi run make-refs` (as nicebot). "
                "Also check the Kerberos ticket (kinit) and that xrdcp is installed."
            )
            sys.exit(2)
        with open(dest) as f:
            ref = counts_from_json(json.load(f))

    ok, table = compare_counts(ref, produced)
    print(f"[{scan_name}] per-step success counts (reference vs produced):")
    print(table)

    if ok:
        log.info(f"[{scan_name}] per-step counts match the reference.")
        sys.exit(0)
    log.error(f"[{scan_name}] per-step counts DIFFER from the reference (possible regression).")
    sys.exit(1)


if __name__ == "__main__":
    main()
