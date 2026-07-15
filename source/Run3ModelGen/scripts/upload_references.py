#!/usr/bin/env python

'''Upload previously-produced reference files to the shared nicebot EOS area.

Run `pixi run make-refs` first to produce + review the scans; then, once you're
happy with the printed summary, `pixi run upload-refs` uploads, for each scan,
run/scan_<name>/ntuple.0.0.root and a counts_<name>.json sidecar (recomputed from
the produced scan) to root://eosuser.cern.ch//eos/user/n/nicebot/Run3ModelGen/.

Credentials come from the environment: export SERVICE_PASS (base64 of the nicebot
password) and CERN_USER (=nicebot). kinit is done into a PRIVATE, auto-destroyed
credential cache, in a new session so it reads the password from stdin (never the
terminal), and neither prints the password nor clobbers your own ticket.

Manual task: `pixi run upload-refs`.
'''

import os
import sys
import json
import base64
import atexit
import tempfile
import subprocess

import click

from Run3ModelGen.refcheck import (
    SCANS, by_name, ntuple_path, ntuple_url, counts_url,
    per_step_counts, format_counts_table, counts_to_json,
)

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)


def _establish_ticket():
    '''kinit for the upload using the SERVICE_PASS/CERN_USER environment vars.

    Reads the base64 password from the environment (never hard-coded/printed),
    decodes it in-process, kinit's into a private credential cache destroyed on
    exit, and runs kinit in a new session (no controlling terminal) so it reads
    the password from stdin instead of prompting the tty.
    '''
    service_pass = os.environ.get("SERVICE_PASS", "")
    cern_user = os.environ.get("CERN_USER", "")
    if not service_pass or not cern_user:
        log.error("SERVICE_PASS (base64 of the nicebot password) and CERN_USER (e.g. nicebot) "
                  "must be set to upload references.")
        sys.exit(2)
    try:
        password = base64.b64decode(service_pass, validate=True)
    except Exception:
        log.error("SERVICE_PASS is not valid base64. Set it to the base64 of the password, e.g.: "
                  "export SERVICE_PASS=$(printf %s 'PASSWORD' | base64 -w0)")
        sys.exit(2)

    # Private credential cache so we don't overwrite the developer's own ticket.
    fd, ccache = tempfile.mkstemp(prefix="krb5cc_makerefs.")
    os.close(fd)
    os.environ["KRB5CCNAME"] = f"FILE:{ccache}"

    def _cleanup():
        subprocess.run(["kdestroy"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            os.remove(ccache)
        except OSError:
            pass
    atexit.register(_cleanup)

    log.info(f"kinit {cern_user}@CERN.CH (into a private, auto-destroyed credential cache)")
    proc = subprocess.run(
        ["kinit", f"{cern_user}@CERN.CH"],
        input=password,               # exact decoded bytes, fed on stdin
        start_new_session=True,       # no controlling tty -> kinit reads stdin, not /dev/tty
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip() or "kinit failed"
        log.error(f"kinit failed: {err}. Check CERN_USER, and that SERVICE_PASS decodes to the "
                  "correct password (verify with: printf '%s' \"$SERVICE_PASS\" | base64 -d).")
        sys.exit(proc.returncode)


def _xrdcp(local_path, url):
    log.info(f"uploading {local_path} -> {url}")
    subprocess.run(["xrdcp", "-f", local_path, url], check=True)


@click.command()
@click.option('--scans', default='all', help="Comma-separated scan names to upload, or 'all'.")
@click.option('--run_dir', default='run', help='Directory containing the produced scans (default: run).')
def main(scans, run_dir):
    '''Upload produced reference files (ntuple + counts JSON) to EOS.'''
    selected = SCANS if scans == 'all' else [by_name(n.strip()) for n in scans.split(',')]

    # Verify all requested references exist before authenticating or uploading anything.
    missing = [s["name"] for s in selected
               if not os.path.exists(ntuple_path(os.path.join(run_dir, s["scan_dir"])))]
    if missing:
        log.error(f"No produced ntuple found for: {', '.join(missing)}. "
                  "Run `pixi run make-refs` first to produce (and review) the references.")
        sys.exit(1)

    _establish_ticket()

    for scan in selected:
        name = scan["name"]
        scan_dir = os.path.join(run_dir, scan["scan_dir"])
        counts = per_step_counts(scan_dir)
        print(format_counts_table(counts, title=f"[{name}] uploading (passed/attempted):"))
        _xrdcp(ntuple_path(scan_dir), ntuple_url(name))
        with tempfile.TemporaryDirectory() as tmp:
            cj = os.path.join(tmp, f"counts_{name}.json")
            with open(cj, 'w') as f:
                json.dump(counts_to_json(counts), f, indent=2)
            _xrdcp(cj, counts_url(name))
        log.info(f"[{name}] uploaded reference ntuple + counts to EOS")

    log.info("Done uploading references.")


if __name__ == "__main__":
    main()
