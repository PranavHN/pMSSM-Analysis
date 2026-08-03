#!/usr/bin/env python

'''Compare a freshly produced scan against its reference on EOS.

Used by the CI genModels jobs: downloads the reference for the given scan from the
shared nicebot EOS area, recomputes the same quantities for the just-produced scan,
prints both, and exits non-zero on any mismatch (which fails the CI job).

Two levels, per :data:`Run3ModelGen.refcheck.SCANS`:

  * per-step counts (all scans) -- how many models cleared each pipeline step;
  * branch VALUES (scans marked ``compare_values``) -- every branch of the ntuple
    compared entry by entry against the reference ntuple, which upload_references.py
    has always published alongside the counts sidecar but nothing used to read.

Counts are the right gate for the random priors, where a value comparison would pin
20 arbitrary sampled points. They are near-vacuous for the mcmc (emcee) chain, where
every model clears every step -- so that scan compares values, which pins the whole
chain trajectory. ``--values/--no-values`` overrides the per-scan default.

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
    counts_from_json, ntuple_url, ntuple_path, compare_ntuples,
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
@click.option('--values/--no-values', 'values', default=None,
              help="Also compare every ntuple branch value against the reference ntuple. "
                   "Default: the scan's `compare_values` setting in refcheck.SCANS.")
def main(scan_name, run_dir, values):
    '''Compare the produced scan against its EOS reference.'''
    scan = by_name(scan_name)
    scan_dir = os.path.join(run_dir, scan["scan_dir"])
    do_values = scan.get("compare_values", False) if values is None else values

    def _fetch(url, dest):
        try:
            subprocess.run(["xrdcp", "-f", url, dest], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log.error(
                f"[{scan_name}] could not fetch reference from {url}: {exc}. "
                "Has it been seeded? Run `pixi run make-refs` (as nicebot). "
                "Also check the Kerberos ticket (kinit) and that xrdcp is installed."
            )
            sys.exit(2)

    # Per-step counts of the just-produced scan.
    produced = per_step_counts(scan_dir)

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        # Reference per-step counts (JSON sidecar).
        dest = os.path.join(tmp, "counts.json")
        _fetch(counts_url(scan_name), dest)
        with open(dest) as f:
            ref = counts_from_json(json.load(f))

        ok_counts, table = compare_counts(ref, produced)
        print(f"[{scan_name}] per-step success counts (reference vs produced):")
        print(table)
        if not ok_counts:
            failures.append("per-step counts")

        # Reference branch values (the ntuple upload_references.py publishes).
        if do_values:
            ref_ntuple = os.path.join(tmp, f"ntuple_{scan_name}.root")
            _fetch(ntuple_url(scan_name), ref_ntuple)
            ok_values, vtable = compare_ntuples(ref_ntuple, ntuple_path(scan_dir))
            print(f"[{scan_name}] ntuple branch values (reference vs produced):")
            print(vtable)
            if not ok_values:
                failures.append("branch values")

    if not failures:
        what = "per-step counts + branch values" if do_values else "per-step counts"
        log.info(f"[{scan_name}] {what} match the reference.")
        sys.exit(0)
    log.error(f"[{scan_name}] {' and '.join(failures)} DIFFER from the reference "
              f"(possible regression).")
    sys.exit(1)


if __name__ == "__main__":
    main()
