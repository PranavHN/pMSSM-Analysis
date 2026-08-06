'''Facade helpers: map the script entrypoints onto snakemake invocations.

The user-facing scripts (genModels.py, extractModels.py, rerunStep.py) share a
--backend local|slurm|htcondor flag selecting the workflow profile; everything
else about their behavior is backend-independent. Cluster backends DETACH the
snakemake orchestrator by default: its log goes to the run log dir, and a killed
orchestrator only PAUSES the run — jobs in flight finish (each cluster job runs
its whole batch group independently), and re-running the same command resumes
where it left off.

Detached is not the same as immortal: run_detached uses start_new_session, which
survives SIGHUP, but systemd-logind still reaps a user's processes when their
last login session ends unless lingering is enabled for the account
(`loginctl enable-linger $USER`, `Linger=yes` in `loginctl show-user $USER`).
Without it, logging off the submit node leaves the already-submitted jobs
running but no driver to schedule the rest.
'''

import getpass
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import time

import yaml

BACKENDS = ("local", "slurm", "htcondor")

# Scheduler binary that must exist on the submit host per backend.
_SCHEDULER_BINARY = {"slurm": "sbatch", "htcondor": "condor_submit"}


def repo_root() -> str:
    '''Repo root from the build env (absolute, baked by cmake_build_env.sh).'''
    return os.environ['RUN3MODELGENDIR'].rstrip('/')


def default_runid() -> str:
    '''Digits-only run identifier (ClusterId-shaped: keeps the scan.<C>.<P>
    naming contract intact). Timestamp-based, overridable via --runid.'''
    return time.strftime('%y%m%d%H%M%S')


def snakemake_argv(backend: str, cfg: dict, targets=(), extra=()) -> list:
    '''Build the snakemake command line for a backend profile + config dict.'''
    repo = repo_root()
    argv = ["snakemake", "-s", f"{repo}/workflow/Snakefile",
            "--workflow-profile", f"{repo}/workflow/profiles/{backend}"]
    argv += list(extra)
    entries = [f"{k}={v}" for k, v in cfg.items() if v is not None]
    if entries:
        argv += ["--config"] + entries
    if targets:
        argv += ["--"] + list(targets)
    return argv


def htcondor_profile_defaults() -> dict:
    '''The htcondor profile's default-resources block, as declared on disk.'''
    cfg = os.path.join(repo_root(), "workflow", "profiles", "htcondor", "config.yaml")
    with open(cfg) as handle:
        return yaml.safe_load(handle).get("default-resources") or {}


def flavour_resources(flavour: str) -> list:
    '''--default-resources entries for a JobFlavour, passed as a COMPLETE set.

    Snakemake's --default-resources REPLACES the profile's default-resources block
    rather than merging with it, so every entry the htcondor profile declares has
    to be repeated on the command line or it silently vanishes for anything
    launched through the entrypoints.

    The values are therefore READ FROM THE PROFILE rather than restated here: the
    profile is the single source of truth and the two cannot drift. They used to be
    two hand-kept lists, and because the pinning test compared only key NAMES, a
    profile-only edit of a VALUE passed the suite while every real submission kept
    using the copy in this file -- silently, since the profile is also the file a
    reader would check. Only classad_JobFlavour is substituted, being per-run.

    NB the core-count override (+RequestCpus) deliberately lives in the profile's
    set-resources, NOT in default-resources: a default-resource is applied to every
    rule and then SUMMED across the widest DAG level, so classad_RequestCpus=2 as a
    default became 5 x 2 = 10 cores -- worse than the 7 it was meant to fix.
    set-resources is not replaced by a CLI --default-resources, so it survives this
    entrypoint untouched and needs no mirroring here.
    '''
    declared = htcondor_profile_defaults()
    # Quoted so snakemake evaluates the value to the *bare* string espresso: the
    # htcondor plugin adds the quotes the submit file needs itself
    # (+JobFlavour = "espresso"), so embedding them here yields ""espresso"".
    entries = [f"classad_JobFlavour='{flavour}'"]
    entries += [f"{key}={value}" for key, value in declared.items()
                if key != "classad_JobFlavour"]
    return entries


def scheduler_extra(backend: str, log_dir: str = None, flavour: str = "tomorrow") -> list:
    '''Extra snakemake args per backend: per-job scheduler logs under
    <log_dir>/outfiles (the slurm plugin otherwise deletes successful-job
    logs), and the JobFlavour resource set on htcondor.'''
    if backend == "slurm":
        extra = []
        if log_dir:
            extra += ["--slurm-logdir", f"{log_dir}/outfiles",
                      "--slurm-keep-successful-logs"]
        return extra
    if backend == "htcondor":
        extra = []
        if log_dir:
            extra += ["--htcondor-jobdir", f"{log_dir}/outfiles"]
        extra += ["--default-resources"] + flavour_resources(flavour)
        return extra
    return []


def run_foreground(argv: list) -> int:
    return subprocess.call(argv)


def linger_warnings() -> list:
    '''Will a detached orchestrator survive you logging off THIS node?

    start_new_session (below) survives SIGHUP, but systemd-logind still reaps a
    user's processes when their last session on the node ends unless the account
    lingers — which is what silently killed detached orchestrators while their
    already-submitted jobs sat in the queue.

    Lingering is per-node: /var/lib/systemd/linger is on the submit node's local
    disk, and `lxplus` is a load-balanced alias, so enabling it once does NOT
    cover the next node you land on. Hence a check at submission time rather
    than a line in the docs.

    Best-effort: no loginctl (macOS, containers, non-systemd) means no opinion.
    '''
    try:
        user = getpass.getuser()
        res = subprocess.run(["loginctl", "show-user", user],
                             capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return []
        if any(line.strip() == "Linger=no" for line in res.stdout.splitlines()):
            return [
                f"lingering is OFF for {user} on {socket.gethostname()}: if you log out of this "
                f"node, systemd kills the detached orchestrator. Jobs already submitted still "
                f"finish their whole batch (ntuple, tarball and publish included), but nothing "
                f"schedules the rest and NOTHING RETRIES A FAILED BATCH — check the run afterwards "
                f"and re-run this exact command to pick up whatever is missing. "
                f"Fix once per submit node:  loginctl enable-linger {user}   "
                f"(or use --foreground inside tmux/screen).",
            ]
    except Exception:
        pass
    return []


def run_detached(argv: list, logfile: str, warn=None) -> int:
    '''Start the orchestrator in its own session; returns the PID.

    `warn` (e.g. log.warning) receives any survivability caveats — see
    linger_warnings(). Kept here so every entrypoint that detaches reports them.
    '''
    if warn is not None:
        for message in linger_warnings():
            warn(message)
    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)

    def _detach():
        # start_new_session already puts the orchestrator in its own session with no
        # controlling terminal, so no TERMINAL-generated signal can reach it. That was
        # not enough: on viper02 the orchestrator was killed by SIGINT during its first
        # couple of seconds -- reproducibly, ~7 of 8 attempts, from a tool AND from an
        # interactive shell -- always inside the slurm executor plugin's module-scope
        # `import pandas`. Every survivor had a parent that outlived that window; every
        # casualty had a parent that exited during it. So the interrupt is aimed at the
        # launcher's process tree and lands on the freshly-spawned child before it is
        # reparented away. An orchestrator that exists precisely to outlive its launcher
        # has no business dying on that, so ignore it. SIGHUP too, for logout.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    with open(logfile, 'ab') as lf:
        proc = subprocess.Popen(argv, stdout=lf, stderr=lf,
                                start_new_session=True, cwd=os.getcwd(),
                                preexec_fn=_detach)
    return proc.pid


# ── pre-flight checks (run BEFORE any wipe/submission side effect) ─────────

def preflight(backend: str, publish: bool = False) -> list:
    '''Return a list of blocking problems (empty = good to go).'''
    problems = []
    binary = _SCHEDULER_BINARY.get(backend)
    if binary and shutil.which(binary) is None:
        problems.append(f"--backend {backend} needs '{binary}' on this host, but it was not found.")
    if publish:
        eospath = os.environ.get('EOSPATH')
        if not eospath:
            problems.append("--publish needs $EOSPATH (source build/setup.sh, or export EOSPATH=...).")
        elif not eospath.startswith("root://"):
            # The publish step mkdir -p's the destination, so requiring existence
            # would be too strict. Walk up to the first component that DOES exist
            # (not the first directory: a file in the way makes mkdir -p fail, so
            # it must be reported rather than skipped over) and check it.
            probe = os.path.abspath(eospath)
            while probe != "/" and not os.path.exists(probe):
                probe = os.path.dirname(probe)
            if not os.path.isdir(probe):
                problems.append(f"--publish destination $EOSPATH={eospath} cannot be created: "
                                f"{probe} exists but is not a directory.")
            elif not os.access(probe, os.W_OK):
                problems.append(f"--publish destination $EOSPATH={eospath} is not writable/creatable "
                                f"on this host (export EOSPATH=... or use a root:// URL).")
    return problems


def locks_active(cwd: str = ".") -> bool:
    '''Best-effort: does another snakemake run appear active from this directory?'''
    lockdir = os.path.join(cwd, ".snakemake", "locks")
    try:
        return bool(os.listdir(lockdir))
    except OSError:
        return False


# ── per-run metadata (batch mode): resume guard replacing --force ──────────

def config_digest(config_file: str, default_config: str) -> str:
    '''sha256 of the effective scan config CONTENT (default config when none given).'''
    path = config_file if config_file else default_config
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_meta_path(meta_dir: str, runid: str) -> str:
    return os.path.join(meta_dir, f"run.{runid}.yaml")


def write_run_meta(meta_dir: str, runid: str, meta: dict) -> str:
    path = run_meta_path(meta_dir, runid)
    os.makedirs(meta_dir, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, default_flow_style=False)
    return path


def _pid_running(pid) -> bool:
    '''Best-effort: is `pid` alive and does it look like a snakemake orchestrator?'''
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return b"snakemake" in f.read()
    except OSError:
        return True  # alive but unreadable: assume it is ours, err on the safe side


def check_run_meta(existing: dict, current: dict) -> tuple:
    '''Compare a re-invocation against the recorded run metadata.

    Returns (fatal_errors, warnings). Fatal: anything that would re-sample and
    thereby wipe finished batches (config content, base_seed), or a still-live
    orchestrator. Warning: num_jobs change (documented top-up/shrink).
    publish/backend changes are allowed silently (add-on targets / same DAG).
    '''
    fatal, warn = [], []
    if existing.get("config_sha256") != current.get("config_sha256"):
        fatal.append("the scan config CONTENT differs from the recorded run — re-submitting would "
                     "re-sample and wipe finished batches. Pick a new --runid for a new config.")
    if existing.get("base_seed") != current.get("base_seed"):
        fatal.append(f"--base_seed {current.get('base_seed')} differs from the recorded "
                     f"{existing.get('base_seed')} — same consequence; pick a new --runid.")
    if _pid_running(existing.get("orchestrator_pid")):
        fatal.append(f"a detached orchestrator for this run appears to be running "
                     f"(pid {existing.get('orchestrator_pid')}); wait for it or kill it first.")
    if existing.get("num_jobs") != current.get("num_jobs"):
        warn.append(f"num_jobs changes from {existing.get('num_jobs')} to {current.get('num_jobs')}: "
                    f"new batches are added with their deterministic seeds / extra dirs are left alone.")
    # Dropping --publish from a run that published WITHOUT --keep_local: the local
    # scan dirs are gone, and without --publish the requested targets become those
    # very ntuples. Snakemake then runs the ntuple rule, which dies in the
    # out-of-sync guard and advises --forceall — i.e. regenerating batches that
    # are in fact complete on EOS. Catch it here, before any of that.
    if existing.get("publish") and not existing.get("keep_local") and not current.get("publish"):
        fatal.append("this run was published without --keep_local, so each batch's local scan dir "
                     "was removed once its EOS copies verified. Re-running without --publish asks "
                     "for those deleted ntuples as targets and fails in the ntuple rule. Add "
                     "--publish to resume/verify this run, or pick a new --runid for an "
                     "unpublished one.")
    return fatal, warn
