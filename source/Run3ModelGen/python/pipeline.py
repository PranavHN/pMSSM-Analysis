'''Shared config/pipeline helpers for the legacy CLI path and the Snakemake workflow.

Single source of truth for:
  * the config merge against the default config (:func:`load_scan_config`),
  * tool availability + platform-conditional step filtering
    (:func:`tool_available`, :func:`resolve_steps`),
  * the per-job model cap (:func:`check_model_cap`),
  * per-batch seed / run-id derivation (:func:`batch_seed`),
  * per-step manifest files used as declared Snakemake rule outputs
    (:func:`write_manifest` / :func:`read_manifest`),
  * the ntuple naming contract (:func:`ntuple_name`, mirroring ntupling.mkntuple).

Logic extracted from ModelGenerator.__init__ (modelgen.py) — ModelGenerator
delegates here, so both execution paths share one implementation.

No environment variables are read at import time.
'''

import hashlib
import json
import os
import shutil
import socket

import yaml

import structlog
log = structlog.get_logger()

# The per-job model cap: larger scans must be split into batches/jobs.
MAX_MODELS_PER_JOB = 10000


def read_yaml_file(file_path: str) -> dict:
    '''Function for reading yaml file. Returns dict.'''
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data


def load_scan_config(config_file: str, default_config: str) -> dict:
    '''Merge a user scan config onto the default config.

    Every key of the default is taken from the user config when present
    (special case: a missing ``num_models`` with ``prior: fixed`` is inferred
    from the per-model parameter lists); extra user keys not present in the
    default (e.g. ``grid``, ``importance``) are carried through unchanged.
    '''
    config_dict = read_yaml_file(config_file) if config_file is not None else {}
    default_dict = read_yaml_file(default_config)

    merged = {}
    for key, defaultval in default_dict.items():
        if key in config_dict:
            merged[key] = config_dict[key]
        # If num_models is not supplied and the prior is 'fixed', read number of models from parameters
        elif key == 'num_models' and config_dict.get('prior', None) == 'fixed':
            merged[key] = len(config_dict['parameters']['tanb'])
        else:
            merged[key] = defaultval

    # Load extra config keys not in default (e.g., 'grid' for grid sampling)
    for key, value in config_dict.items():
        if key not in default_dict:
            merged[key] = value

    return merged


# Steps that exist only when an external binary does; every other step always runs.
STEP_EXECUTABLES = {'feynhiggs': 'FeynHiggs'}


def env_bin_dir() -> str:
    '''The pixi default environment's bin directory, or '' if not derivable.

    Read from RUN3MODELGENDIR (exported by build/setup.sh) at CALL time, never
    at import (see the module docstring).
    '''
    repo = os.environ.get('RUN3MODELGENDIR', '').rstrip('/')
    return f"{repo}/.pixi/envs/default/bin" if repo else ''


def tool_available(exe: str, env_bin: str = None) -> bool:
    '''Is `exe` usable by a pipeline step? Checks the pixi env AND PATH.

    The union is what makes the two callers agree. They ask from different
    environments: the Snakefile resolves the DAG at PARSE time, where PATH need
    not be set up yet (remote executors re-parse the Snakefile inside every job,
    before the shell prefix runs), so it can only look in the env by absolute
    path; ModelGenerator resolves at RUN time, after build/setup.sh has put the
    env on PATH, and must also see a tool installed outside the env. Either
    oracle alone therefore answers differently in one direction or the other —
    and a disagreement is silent: the DAG grows a step whose body then refuses
    it ("step not in resolved steps"), or drops one the body would have run.

    `env_bin` overrides the directory (the Snakefile passes its own ENV_BIN,
    derived from workflow.basedir rather than from the environment).
    '''
    bin_dir = env_bin_dir() if env_bin is None else env_bin
    if bin_dir and os.path.exists(f"{bin_dir}/{exe}"):
        return True
    return shutil.which(exe) is not None


def resolve_steps(steps: list, env_bin: str = None) -> list:
    '''Drop pipeline steps whose external tool is not installed on this platform.

    In particular FeynHiggs is not packaged on conda-forge for macOS, so on that
    platform the 'feynhiggs' step is silently removed rather than failing every
    model (and thereby skipping all downstream steps). On Linux, where FeynHiggs
    is installed, this is a no-op.

    Mirrored at parse time by _resolve_steps in the Snakefile — change both
    together; tests/test_snakefile_mirrors.py pins the agreement.
    '''
    available_steps = []
    for step in steps:
        exe = STEP_EXECUTABLES.get(step['name'])
        if exe is not None and not tool_available(exe, env_bin):
            log.warning(f"Skipping step '{step['name']}': required executable "
                        f"'{exe}' found neither in the pixi environment nor on "
                        f"PATH (not available on this platform).")
            continue
        available_steps.append(step)
    return available_steps


def check_model_cap(num_models: int) -> None:
    '''Enforce the per-job model cap (split larger scans into multiple jobs).'''
    if num_models > MAX_MODELS_PER_JOB:
        log.fatal(f"num_models too large! Please reduce to max 10,000 and split into multiple jobs using genModels.py --num_jobs N!")
        exit()


# ── Batch identity ─────────────────────────────────────────────────────────

def batch_seed(runid: str, base_seed: int, b: int) -> int:
    '''Deterministic per-batch numpy seed.

    Replaces the legacy ``${ClusterId:6}${ProcId}`` (unknowable pre-submission,
    unreproducible): (runid, base_seed, b) fully determine every batch's sampled
    points. Hashing decorrelates neighbouring batches; the result fits numpy's
    legacy 32-bit seed bound. Recorded in scan_config.yaml / run_meta.yaml.
    '''
    h = hashlib.blake2b(f"{runid}:{base_seed}:{b}".encode(), digest_size=4)
    return int.from_bytes(h.digest(), "big")


def ntuple_name(scan_dir: str) -> str:
    '''Path of the ntuple mkntuple() will write for a scan dir (naming contract).

    Mirrors ntupling.mkntuple: a ``scan.<ClusterId>.<ProcId>`` basename yields
    ``ntuple.<ClusterId>.<ProcId>.root``, anything else ``ntuple.0.0.root``.
    '''
    scan_dir = scan_dir.rstrip('/')
    parts = scan_dir.split('/')[-1].split('.')
    if len(parts) == 3:
        return f"{scan_dir}/ntuple.{parts[1]}.{parts[2]}.root"
    return f"{scan_dir}/ntuple.0.0.root"


# ── Which batch job ran this? (recorded, never used for naming) ────────────

def _condor_job_ad(path: str) -> dict:
    '''ClusterId/ProcId/GlobalJobId out of an HTCondor job ad file.'''
    wanted = ("ClusterId", "ProcId", "GlobalJobId")
    out = {}
    with open(path) as f:
        for line in f:
            key, sep, value = line.partition("=")
            if sep and key.strip() in wanted:
                out[key.strip()] = value.strip().strip('"')
    return out


def scheduler_identity() -> dict:
    '''Best-effort identity of the batch job executing this rule.

    Recorded in the per-step manifests so "batch 17 failed" maps to a scheduler
    job id directly. Without it that mapping exists only in the orchestrator
    log, and only as ClusterId <-> group UUID — recovering the batch index means
    recomputing uuid5("publish:b:17") and grepping for it.

    Deliberately NOT fed back into any name or seed: the legacy scheme derived
    both from ${ClusterId}${ProcId}, which is unknowable before submission and
    changes on every resubmission (see batch_seed). This is a record, not an
    identifier.

    Never raises — metadata must not be able to fail a rule.
    '''
    ident = {}
    try:
        ident["host"] = socket.gethostname()
        # SLURM_JOB_ID is current, SLURM_JOBID the older spelling; both appear
        # in the wild depending on the site's slurm version.
        slurm_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
        if slurm_id:
            ident["backend"] = "slurm"
            ident["job_id"] = slurm_id
            # a requeued job (slurm-requeue: true in the profile) keeps its id but
            # gets a new restart count, which is what distinguishes the attempts
            if os.environ.get("SLURM_RESTART_COUNT"):
                ident["restart_count"] = os.environ["SLURM_RESTART_COUNT"]
        elif os.environ.get("_CONDOR_JOB_AD", "") and os.path.isfile(os.environ["_CONDOR_JOB_AD"]):
            ad = _condor_job_ad(os.environ["_CONDOR_JOB_AD"])
            ident["backend"] = "htcondor"
            if "ClusterId" in ad and "ProcId" in ad:
                # the form condor_q/condor_rm take
                ident["job_id"] = f"{ad['ClusterId']}.{ad['ProcId']}"
            if "GlobalJobId" in ad:
                ident["global_job_id"] = ad["GlobalJobId"]
        else:
            ident["backend"] = "local"
    except Exception:
        pass
    return ident


# ── Per-step manifests (declared Snakemake outputs; failure-is-data) ───────

def write_manifest(path: str, step: str, num_models: int,
                   attempted: list, succeeded: list, chain_ok: list) -> None:
    '''Write a per-step manifest: which models were attempted/succeeded, and
    which are still alive for the next step under the legacy list-order gating
    (a model is dropped from the chain on its first failing step).'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {
        "step": step,
        "num_models": num_models,
        "attempted": sorted(attempted),
        "succeeded": sorted(succeeded),
        "chain_ok": sorted(chain_ok),
        "counts": [len(succeeded), num_models],
        # which batch job produced this step (record only — see scheduler_identity)
        "scheduler": scheduler_identity(),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, path)


def read_manifest(path: str) -> dict:
    '''Read a per-step manifest written by :func:`write_manifest`.'''
    with open(path) as f:
        return json.load(f)
