#!/usr/bin/env python
'''Internal per-batch CLI driving the Snakemake workflow rules.

Each subcommand is the body of ONE Snakemake rule for ONE batch (= one scan
dir): sample | step | ntuple | chain | publish. Not intended for direct user
consumption — genModels.py / extractModels.py / rerunStep.py are the
user-facing entrypoints.

Requires the usual environment (source build/setup.sh): DATAPATH etc. at
import, tools on PATH.

Failure semantics: per-model tool failure is DATA (recorded in the manifest,
exit 0). A non-zero exit means an infrastructure error (missing files, broken
env) and makes the Snakemake rule fail/retry.
'''

import errno
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported here, so the emcee worker
# Pool (one process per core) does not oversubscribe via nested BLAS
# threads. genModels.py is now a thin Snakemake facade that never imports numpy/forks,
# so the pin must live in this process (where the pipeline imports numpy). setdefault
# lets an explicit sbatch/profile export win.
for _blas_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import click
import numpy as np

from Run3ModelGen import pipeline
from Run3ModelGen.modelgen import ModelGenerator, datadir
from Run3ModelGen import steps as toolsteps

import structlog
log = structlog.get_logger()

# The one step whose in-place xsec injection must not mutate the upstream
# SPheno output under Snakemake: it gets an xsec-augmented COPY in this dir
# instead. The EOS packaging folds the copy back over SPheno/ inside the
# tarball, so published tarballs are unchanged (see doc/snakemake.md).
XSEC_DIR = "SModelS_xsec"


def _stage_file(src: str, dst: str) -> None:
    '''Hardlink src to dst, falling back to a copy where links are refused.

    Staging the tar tree with links avoids copying the scan's data, but AFS
    (the lxplus/HTCondor scan_parent) rejects hardlinks BETWEEN DIRECTORIES
    even inside one volume — os.link then raises EXDEV, and legacy scans never
    hit it only because they ran on the worker's local /pool/condor disk. The
    tar contents are identical either way, so a copy is a pure fallback.
    '''
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP):
            raise
        shutil.copy2(src, dst)


def _load_generator(config_file: str, scan_dir: str, seed: int, quiet: bool = False) -> ModelGenerator:
    '''Build a ModelGenerator without touching the scan dir contents.

    quiet=True for the per-step/ntuple rules so the logo + config dump appear
    once per batch (in the sample/chain rule), not once per rule.'''
    return ModelGenerator(config_file=config_file or None, scan_dir=scan_dir, seed=seed, quiet=quiet)


def _restore_points(mg: ModelGenerator, points_file: str) -> None:
    '''Load the sampled points + final num_models from the sample rule's npz.'''
    data = np.load(points_file)
    mg.num_models = int(data["_num_models"])
    for key in data.files:
        if not key.startswith("_"):
            mg.points[key] = data[key]


@click.group()
def cli():
    '''Per-batch workflow rule bodies (internal).'''


@cli.command()
@click.option('--config_file', default=None, help='Scan config yaml (None = default config).')
@click.option('--scan_dir', required=True, type=str, help='Batch scan directory.')
@click.option('--seed', required=True, type=int, help='numpy seed for this batch.')
@click.option('--points_out', required=True, type=str, help='Output .npz with sampled points.')
def sample(config_file, scan_dir, seed, points_out):
    '''Sample all points for one batch and set up a FRESH scan dir.

    Wipes any pre-existing scan dir (legacy fresh-scan semantics): Snakemake
    only re-runs this rule when config/seed/code changed, in which case every
    downstream step re-runs anyway.
    '''
    mg = _load_generator(config_file, scan_dir, seed)

    # Fresh scan dir (mirrors generate_models' legacy wipe). Guarded: refuses to
    # destroy an emcee chain.h5 (see pipeline.wipe_scan_dir) and logs the path.
    pipeline.wipe_scan_dir(mg.scan_dir)
    mg._setup_scan_dirs()

    if mg.prior == 'flat':
        mg.sample_flat()
    elif mg.prior == 'fixed':
        mg.sample_fixed()
    elif mg.prior == 'grid':
        mg.sample_grid()
    elif mg.prior == 'importance':
        mg.sample_importance()
    else:
        raise ValueError(f"prior {mg.prior} is not point-sampled (mcmc runs as a chain rule)")

    mg._dump_scan_config()

    os.makedirs(os.path.dirname(os.path.abspath(points_out)), exist_ok=True)
    np.savez(points_out, _num_models=mg.num_models, **mg.points)
    log.info(f"Sampled {mg.num_models} points for {scan_dir} (seed {seed})")


@cli.command()
@click.option('--step', 'step_name', required=True, type=str, help='Pipeline step name.')
@click.option('--config_file', default=None)
@click.option('--scan_dir', required=True, type=str)
@click.option('--points', 'points_file', required=True, type=str)
@click.option('--prev_manifest', default=None, help='Manifest of the previous step (None for the first step).')
@click.option('--manifest_out', required=True, type=str)
def step(step_name, config_file, scan_dir, points_file, prev_manifest, manifest_out):
    '''Run ONE tool over one batch, gated by the previous step's manifest.

    Gating replicates the legacy list-order semantics (modelgen.py
    generate_models): a model is attempted here only if EVERY earlier step in
    the config order succeeded — so per-step success counts (the CI reference
    contract) match the legacy loop exactly.
    '''
    # Absolute paths first: tools with cwd-relative scratch run in a private tmpdir.
    scan_dir = os.path.abspath(scan_dir)
    points_file = os.path.abspath(points_file)
    manifest_out = os.path.abspath(manifest_out)
    if config_file:
        config_file = os.path.abspath(config_file)
    if prev_manifest:
        prev_manifest = os.path.abspath(prev_manifest)

    mg = _load_generator(config_file, scan_dir, seed=0, quiet=True)  # seed unused: no sampling here
    _restore_points(mg, points_file)

    step_cfg = next((s for s in mg.steps if s['name'] == step_name), None)
    if step_cfg is None:
        raise click.ClickException(f"step '{step_name}' not in resolved steps")

    if prev_manifest:
        gate = pipeline.read_manifest(prev_manifest)["chain_ok"]
    else:
        gate = list(range(mg.num_models))

    # Consistency guard: the per-model files are deliberately NOT declared
    # snakemake outputs (failed models have none), so a manually deleted scan
    # dir goes unnoticed by the DAG while the manifests still say "done".
    # Refuse loudly instead of recording every model as a physics failure.
    input_dir = step_cfg.get('input_dir')
    if input_dir and gate:
        missing = [m for m in gate if not os.path.exists(f"{scan_dir}/{input_dir}/{m}.slha")]
        if missing:
            raise click.ClickException(
                f"scan dir out of sync with its manifests: {len(missing)}/{len(gate)} input files "
                f"missing from {scan_dir}/{input_dir} (e.g. {missing[0]}.slha) although the previous "
                f"step's manifest records them as produced. If the scan dir was deleted, also remove "
                f"its bookkeeping (rm -rf <scan_parent>/r3mg/<scan>) or re-run with --forceall to "
                f"regenerate everything.")

    if step_name == 'SModelS':
        os.makedirs(f"{scan_dir}/{XSEC_DIR}", exist_ok=True)

    # Private working directory: some tools drop scratch files into the cwd
    # (micrOMEGAs/CalcHEP, SPheno's Messages.out); concurrent batch jobs must
    # not share one.
    workdir = tempfile.mkdtemp(prefix=f"r3mg_{step_name}_")
    os.chdir(workdir)

    succeeded, usable = [], []
    for mod in gate:
        log.info(f"\trunning step: {step_name} (model {mod})")
        if step_name == 'SModelS':
            # Immutable-SPheno mode: xsecs go to a copy, not into SPheno/<n>.slha
            infile = f"{scan_dir}/{step_cfg['input_dir']}/{mod}.slha"
            ok = toolsteps.run_SModelS(
                infile=infile,
                output_dir=f"{scan_dir}/{step_cfg['output_dir']}/",
                smodels_ini=f"{datadir}/SModelS_parameters.ini",
                xsec_dest=f"{scan_dir}/{XSEC_DIR}/{mod}.slha")
        else:
            ok = mg.run_step(modelnum=mod, **step_cfg)
        if ok:
            succeeded.append(mod)
            # The reported success rate is "produced variables the ntuple can carry",
            # which is not the same as "the tool exited 0": SModelS writes a result
            # file with no ExptRes when it had no cross sections, and SPheno's
            # charged-LSP SPINFO warning makes the spectrum unreadable. Counted with
            # the ntuple's own extraction so the number cannot drift from reality.
            if mg.step_produced_ntuple_vars(mod, step_cfg):
                usable.append(mod)

    # `chain_ok` stays the tool-ran set: it is what the next step gates on, and a
    # stricter statistic must not silently shrink the dataset.
    pipeline.write_manifest(manifest_out, step=step_name, num_models=mg.num_models,
                            attempted=gate, succeeded=succeeded, chain_ok=succeeded,
                            with_ntuple_vars=usable)
    lost = len(succeeded) - len(usable)
    note = f"  [{lost} ran but produced NO ntuple variables]" if lost else ""
    log.info(f"{step_name}: {len(usable)}/{len(succeeded)}/{mg.num_models} "
             f"(with ntuple vars / tool ran / total; attempted {len(gate)}){note}")


@cli.command()
@click.option('--config_file', default=None)
@click.option('--scan_dir', required=True, type=str)
@click.option('--points', 'points_file', required=True, type=str)
@click.option('--meta_dir', default=None, type=str, help='Manifest dir (default: the directory of --points).')
def ntuple(config_file, scan_dir, points_file, meta_dir):
    '''Gather all step outputs of one batch into the ROOT ntuple.

    First re-prints the per-step success summary from the step manifests —
    the batch-level equivalent of the legacy end-of-run stats block.'''
    mg = _load_generator(config_file, scan_dir, seed=0, quiet=True)
    _restore_points(mg, points_file)

    # Consistency guard (see step()): a wiped scan dir with intact manifests
    # would otherwise yield an all-sentinel ntuple without any error.
    if not os.path.exists(f"{scan_dir}/scan_config.yaml"):
        raise click.ClickException(
            f"scan dir {scan_dir} is missing scan_config.yaml although its manifests exist — "
            f"the scan dir was deleted or moved. Remove its bookkeeping "
            f"(rm -rf <scan_parent>/r3mg/<scan>) or re-run with --forceall to regenerate.")

    meta_dir = meta_dir or os.path.dirname(os.path.abspath(points_file))
    log.info("Finished generating models! Stats for successful generations (successful/attempted):")
    for step_cfg in mg.steps:
        mpath = os.path.join(meta_dir, f"step_{step_cfg['name']}.json")
        if not os.path.exists(mpath):
            continue
        m = pipeline.read_manifest(mpath)
        log.info(f"\t{m['step']}: {m['counts'][0]}/{m['counts'][1]} (attempted {len(m['attempted'])})")

    mg.mkntuple()


@cli.command()
@click.option('--config_file', default=None)
@click.option('--scan_dir', required=True, type=str)
@click.option('--seed', required=True, type=int)
@click.option('--manifest_out', required=True, type=str)
@click.option('--nworkers', default=None, type=int,
              help='emcee worker-pool size (defaults to the mcmc.nworkers config).')
def chain(config_file, scan_dir, seed, manifest_out, nworkers):
    '''Run one MCMC chain scan (prior: mcmc) as a single opaque job.

    prior: mcmc runs the emcee affine-invariant ensemble sampler
    (Run3ModelGen.emcee_sampler): the whole pipeline runs inside sampling (so it
    can't be decomposed into per-tool rules), checkpointing to <scan_dir>/chain.h5
    and auto-resuming. The chain owns its scan dir but deliberately does NOT wipe
    it, so re-running the rule resumes the chain (chain.h5 is undeclared side-state;
    see chain.smk).
    '''
    scan_dir = os.path.abspath(scan_dir)
    manifest_out = os.path.abspath(manifest_out)
    if config_file:
        config_file = os.path.abspath(config_file)

    # Private cwd for tool scratch (see step()); removed in the finally so repeated
    # resume runs of the chain rule don't leak a tempdir per invocation.
    orig_cwd = os.getcwd()
    workdir = tempfile.mkdtemp(prefix="r3mg_chain_")
    os.chdir(workdir)
    try:
        mg = _load_generator(config_file, scan_dir, seed)
        if mg.prior != 'mcmc':
            raise click.ClickException(f"chain rule invoked with prior '{mg.prior}'")

        from Run3ModelGen.emcee_sampler import run_emcee_scan
        ecfg = getattr(mg, 'mcmc', {}) or {}
        mg._setup_scan_dirs()                 # non-destructive setup (no wipe -> resume-safe)
        backend_path = os.path.join(scan_dir, 'chain.h5')
        summary = run_emcee_scan(
            config_file=config_file, scan_dir=scan_dir, backend_path=backend_path,
            seed=seed, nworkers=nworkers,
            max_hours=(ecfg.get('walltime_hours') or 23.0),
            resume=os.path.exists(backend_path),
            to_ntuple=pipeline.ntuple_name(scan_dir),
            worker_base=workdir)
        mg._dump_scan_config()
        manifest = {"engine": "emcee", "seed": seed,
                    "iterations": summary.get("iterations"),
                    "n_samples": summary.get("n_samples"),
                    "converged": summary.get("converged"),
                    "mean_acceptance": summary.get("mean_acceptance"),
                    "max_tau": summary.get("max_tau"),
                    "scheduler": pipeline.scheduler_identity()}

        os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
        with open(manifest_out, "w") as f:
            json.dump(manifest, f, indent=1)

        # Final step: emcee diagnostics -> <scan_dir>/diag (a declared directory output of
        # the chain rule). Best-effort — a plotting/arviz hiccup must NOT fail a completed
        # chain — but the dir is always created so the declared output exists.
        diag_dir = os.path.join(scan_dir, 'diag')
        os.makedirs(diag_dir, exist_ok=True)
        try:
            rc = subprocess.run(['emcee_diagnostics.py', '--glob', backend_path,
                                 '--outdir', diag_dir]).returncode
            if rc != 0:
                log.warning(f"emcee diagnostics exited {rc} (non-fatal); partial output in {diag_dir}")
        except Exception as e:
            log.warning(f"emcee diagnostics failed (non-fatal): {e!r}")
    finally:
        os.chdir(orig_cwd)
        shutil.rmtree(workdir, ignore_errors=True)


@cli.command()
@click.option('--config_file', default=None)
@click.option('--scan_dir', required=True, type=str)
@click.option('--tar_out', required=True, type=str)
def tarball(config_file, scan_dir, tar_out):
    '''Tar one batch scan dir with the legacy member layout.

    Stages the scan tree (see _stage_file), folds each xsec-augmented SModelS_xsec/
    copy back over its SPheno/ member path, and drops the SModelS_xsec dir
    itself — reproducing the tarballs of the legacy in-place-mutation flow
    byte-for-byte (member list AND per-file contents).
    '''
    scan_dir = os.path.abspath(scan_dir).rstrip("/")
    tar_out = os.path.abspath(tar_out)
    name = os.path.basename(scan_dir)

    # Where do the xsec-augmented copies fold back to? The SModelS step's
    # input_dir (the SPheno output dir) from the merged config.
    cfg = pipeline.load_scan_config(config_file or None, f"{datadir}/configs/default_config.yaml")
    spheno_dir = next((s.get('input_dir') for s in cfg.get('steps', [])
                       if s['name'] == 'SModelS'), None)

    # Stage next to the scan dir, so a rename/link stays within one filesystem.
    stage = tempfile.mkdtemp(prefix=".r3mg_stage_", dir=os.path.dirname(scan_dir))
    try:
        dst_root = os.path.join(stage, name)
        for root, dirs, files in os.walk(scan_dir):
            rel = os.path.relpath(root, scan_dir)
            if rel == XSEC_DIR or rel.startswith(f"{XSEC_DIR}{os.sep}"):
                continue
            os.makedirs(os.path.join(dst_root, rel), exist_ok=True)
            for fn in files:
                # the workflow's DAG snapshot (copied into the scan dir at the
                # end of a previous run) is not part of the legacy tar layout
                if rel == "." and fn in ("dag.dot", "dag.svg"):
                    continue
                _stage_file(os.path.join(root, fn), os.path.join(dst_root, rel, fn))

        xdir = os.path.join(scan_dir, XSEC_DIR)
        if spheno_dir and os.path.isdir(xdir):
            for fn in os.listdir(xdir):
                tgt = os.path.join(dst_root, spheno_dir, fn)
                if os.path.exists(tgt):
                    os.remove(tgt)
                _stage_file(os.path.join(xdir, fn), tgt)

        subprocess.run(["tar", "-czf", tar_out, "-C", stage, name], check=True)
        log.info(f"Wrote {tar_out}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _remote_size(url: str):
    '''Byte size of an xrootd URL, or None if it cannot be determined.'''
    parts = re.match(r"(root://[^/]+/)(/.*)", url)
    if not parts:
        return None
    res = subprocess.run(["xrdfs", parts.group(1), "stat", parts.group(2)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        return None
    for line in res.stdout.splitlines():
        if line.strip().startswith("Size:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _unverified(pairs: list) -> list:
    '''Which of the (src, dst) copies cannot be confirmed present and complete?

    Fails CLOSED: a destination whose size cannot be read counts as unverified,
    because the caller deletes the only other copy of the data on an empty list.
    '''
    problems = []
    for src, dst in pairs:
        want = os.path.getsize(src)
        if dst.startswith("root://"):
            got = _remote_size(dst)
        else:
            got = os.path.getsize(dst) if os.path.isfile(dst) else None
        if got is None:
            problems.append(f"{dst}: missing, or its size could not be read")
        elif got != want:
            problems.append(f"{dst}: {got} bytes, expected {want}")
    return problems


@cli.command()
@click.option('--scan_dir', required=True, type=str)
@click.option('--tarball', 'tar_file', required=True, type=str)
@click.option('--runid', required=True, type=str)
@click.option('--procid', required=True, type=str)
@click.option('--marker', default=None, type=str,
              help='Completion marker, written once the copies verify and BEFORE any local delete.')
@click.option('--cleanup', is_flag=True,
              help='Delete the local scan dir + tarball once the published copies verify.')
def publish(scan_dir, tar_file, runid, procid, marker, cleanup):
    '''Copy tarball + config yaml + ntuple to $EOSPATH/<runid>/<procid>/.

    Byte-mirror of the historical condor worker's copy step (git history:
    data/genmodels_condor.sh).
    Plain cp for a mounted $EOSPATH (the legacy lxplus flow); `xrdcp -p -f`
    when $EOSPATH is a root:// URL.

    With --cleanup the batch's local scan dir and tarball are removed afterwards,
    so a published run occupies no space outside $EOSPATH — the legacy disk
    profile, where the worker's scratch dir vanished with the job (a 300-batch
    production run keeps ~340 GB of scan dirs otherwise). Every published file
    is re-checked for presence AND byte size first; nothing is deleted if any
    check fails. The bookkeeping under r3mg/ is small and always kept, so the
    run stays resumable and its per-step counts inspectable.
    '''
    eospath = os.environ['EOSPATH'].rstrip("/")
    scan_dir = os.path.abspath(scan_dir).rstrip("/")
    tar_file = os.path.abspath(tar_file)
    ntuple = pipeline.ntuple_name(scan_dir)
    yamls = sorted(glob.glob(f"{scan_dir}/*.yaml"))

    # One (src, dst) list for both transports: also what the verification reads.
    pairs = ([(tar_file, f"{eospath}/{runid}/{procid}/{os.path.basename(tar_file)}"),
              (ntuple, f"{eospath}/{runid}/{procid}/{os.path.basename(ntuple)}")]
             + [(y, f"{eospath}/{runid}/{os.path.basename(y)}") for y in yamls])

    if eospath.startswith("root://"):
        for src, dst in pairs:
            subprocess.run(["xrdcp", "-p", "-f", src, dst], check=True)
    else:
        os.makedirs(f"{eospath}/{runid}/{procid}", exist_ok=True)
        for src, dst in pairs:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    log.info(f"Published batch {procid} to {eospath}/{runid}/{procid}")

    if cleanup:
        problems = _unverified(pairs)
        if problems:
            raise click.ClickException(
                "refusing to delete the local copies: the published files did not verify.\n  "
                + "\n  ".join(problems)
                + f"\nThe scan dir {scan_dir} and its tarball were left untouched.")

    # Record success BEFORE deleting anything. Dying after the delete but before
    # the marker would strand the batch: its local inputs are gone, so a resume
    # cannot rebuild the ntuple, and without the marker it cannot skip it either.
    # Dying before the marker instead leaves the local copies intact and the
    # batch simply re-publishes on the next attempt (cp/xrdcp -f overwrite).
    if marker:
        os.makedirs(os.path.dirname(os.path.abspath(marker)), exist_ok=True)
        with open(marker, "w"):
            pass

    if not cleanup:
        return

    freed = sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(scan_dir) for f in files)
    freed += os.path.getsize(tar_file) if os.path.isfile(tar_file) else 0
    shutil.rmtree(scan_dir, ignore_errors=True)
    if os.path.isfile(tar_file):
        os.remove(tar_file)
    log.info(f"Verified {len(pairs)} published files; removed the local scan dir "
             f"and tarball ({freed / 1e9:.2f} GB freed). The published copies in "
             f"{eospath}/{runid}/ are now the only ones.")


if __name__ == "__main__":
    cli()
