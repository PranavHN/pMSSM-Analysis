'''Every path an entrypoint hands to snakemake must be absolute.

A cluster worker does NOT run in the submit directory: HTCondor gives the job
its own scratch dir (ShouldTransferFiles=YES), while snakemake is told the
filesystem is shared (--shared-fs-usage ... input-output) and therefore stages
nothing. The two only agree when every path is absolute. A relative --scan_dir
resolved against the worker's scratch instead, so a whole scan — and the r3mg
bookkeeping that makes the run resumable — was built where the orchestrator
could not see it, and the submit host saw an empty directory for hours.

Caught on a real 10,000-model lxbatch run; batch mode was immune only because
it already abspath'd its parent. These tests pin every entrypoint.
'''

import importlib.util
import os

import pytest
import yaml
from click.testing import CliRunner


def _load_script(repo, name):
    '''Import an entrypoint from scripts/ by path (they are CLIs, not modules).'''
    path = repo / "source" / "Run3ModelGen" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def genmodels(repo):
    return _load_script(repo, "genModels.py")


@pytest.fixture(scope="module")
def rerunstep(repo):
    return _load_script(repo, "rerunStep.py")


@pytest.fixture
def captured_argv(monkeypatch, repo):
    '''Run an entrypoint without launching snakemake; capture its command line.'''
    seen = {}
    # snakemake_argv resolves the workflow from the build environment
    monkeypatch.setenv("RUN3MODELGENDIR", str(repo))

    def _capture(argv, *args, **kwargs):
        seen["argv"] = argv
        return 0

    def _install(mod):
        monkeypatch.setattr(mod.snakefacade, "run_foreground", _capture)
        monkeypatch.setattr(mod.snakefacade, "run_detached",
                            lambda argv, logfile, **kw: (seen.update(argv=argv), 4242)[1])
        return seen

    return _install


def _config_value(argv, key):
    '''The value of a `key=value` entry in the --config block.'''
    for entry in argv:
        if entry.startswith(f"{key}="):
            return entry.split("=", 1)[1]
    return None


# ── genModels.py ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", ["local", "htcondor"])
def test_single_mode_passes_an_absolute_scan_dir(genmodels, captured_argv, backend,
                                                 tmp_path, monkeypatch):
    seen = captured_argv(genmodels)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genmodels.snakefacade, "preflight", lambda *a, **k: [])

    res = CliRunner().invoke(genmodels.main, ["--scan_dir", "relscan", "--backend", backend])

    assert res.exit_code == 0, res.output
    scan_dir = _config_value(seen["argv"], "scan_dir")
    assert os.path.isabs(scan_dir), f"relative scan_dir would resolve on the worker: {scan_dir}"
    assert scan_dir == str(tmp_path / "relscan")


def test_batch_mode_passes_an_absolute_scan_parent(genmodels, captured_argv,
                                                   tmp_path, monkeypatch):
    seen = captured_argv(genmodels)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genmodels.snakefacade, "preflight", lambda *a, **k: [])

    res = CliRunner().invoke(genmodels.main, ["--scan_dir", "relparent", "--num_jobs", "2",
                                              "--runid", "90009", "--backend", "htcondor"])

    assert res.exit_code == 0, res.output
    assert os.path.isabs(_config_value(seen["argv"], "scan_parent"))


def test_an_absolute_scan_dir_is_passed_through_unchanged(genmodels, captured_argv,
                                                          tmp_path, monkeypatch):
    seen = captured_argv(genmodels)
    monkeypatch.setattr(genmodels.snakefacade, "preflight", lambda *a, **k: [])
    target = tmp_path / "somewhere" / "scan"

    res = CliRunner().invoke(genmodels.main, ["--scan_dir", str(target)])

    assert res.exit_code == 0, res.output
    assert _config_value(seen["argv"], "scan_dir") == str(target)


def test_a_trailing_slash_does_not_change_the_scan_dir(genmodels, captured_argv,
                                                       tmp_path, monkeypatch):
    '''The basename drives the ntuple name, so "scan/" must not become "".'''
    seen = captured_argv(genmodels)
    monkeypatch.setattr(genmodels.snakefacade, "preflight", lambda *a, **k: [])

    res = CliRunner().invoke(genmodels.main, ["--scan_dir", f"{tmp_path}/scan/"])

    assert res.exit_code == 0, res.output
    assert _config_value(seen["argv"], "scan_dir") == f"{tmp_path}/scan"


# ── genModels.py: where an htcondor batch run puts its data ────────────────
#
# A scan dir is ~1 GB per 10k-model batch. Pointing N of them at the submit volume
# is what ran a 300-batch run out of AFS quota mid-flight: every job died in
# HTCondor's output transfer with "(errno 122) Disk quota exceeded", was held, and
# was reaped 24h later — 300 batches of compute for nothing. So htcondor batch runs
# default the DATA to per-job scratch and keep only the bookkeeping behind.

@pytest.fixture
def batch_env(genmodels, captured_argv, tmp_path, monkeypatch):
    '''A batch invocation that reaches snakemake without touching a scheduler.'''
    seen = captured_argv(genmodels)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(genmodels.snakefacade, "preflight", lambda *a, **k: [])
    monkeypatch.setenv("EOSPATH", str(tmp_path / "eos"))
    return seen


_BATCH = ["--num_jobs", "2", "--runid", "90009"]


def test_htcondor_batch_defaults_the_data_to_condor_scratch(genmodels, batch_env):
    '''UNEXPANDED on purpose: HTCondor only defines _CONDOR_SCRATCH_DIR inside the
    job, so the workflow expandvars-resolves it when each worker re-parses the
    Snakefile. Expanding it here would resolve to the submit host's environment
    (i.e. to nothing) and put every batch back on shared storage.'''
    res = CliRunner().invoke(genmodels.main,
                            _BATCH + ["--backend", "htcondor", "--publish"])

    assert res.exit_code == 0, res.output
    assert _config_value(batch_env["argv"], "scan_parent") == "$_CONDOR_SCRATCH_DIR/scan"


def test_htcondor_batch_keeps_its_bookkeeping_on_shared_storage(genmodels, batch_env, tmp_path):
    '''published.ok, run.<runid>.yaml and the logs are read by the orchestrator on
    the submit host, so they must NOT follow the data onto a worker's scratch — the
    run would lose track of every finished batch. They stay where a shared-FS run
    would have put them, which is what keeps resume working unchanged.'''
    res = CliRunner().invoke(genmodels.main,
                            _BATCH + ["--backend", "htcondor", "--publish"])

    assert res.exit_code == 0, res.output
    meta_parent = _config_value(batch_env["argv"], "meta_parent")
    assert os.path.isabs(meta_parent), f"the orchestrator has to reach it: {meta_parent}"
    assert meta_parent == str(tmp_path / "scan")


def test_htcondor_batch_without_publish_is_refused(genmodels, batch_env):
    '''Scratch is destroyed when the job exits, so a run whose results never leave
    the node produces nothing at all. Refuse instead of burning the cluster time.'''
    res = CliRunner().invoke(genmodels.main, _BATCH + ["--backend", "htcondor"])

    assert res.exit_code != 0
    assert "_CONDOR_SCRATCH_DIR" in res.output
    assert "--publish" in res.output


def test_htcondor_batch_refuses_keep_local_on_scratch(genmodels, batch_env):
    '''--keep_local would also put the ntuples back in the target list at paths that
    only ever exist on a worker, and the orchestrator would wait for them forever.'''
    res = CliRunner().invoke(
        genmodels.main, _BATCH + ["--backend", "htcondor", "--publish", "--keep_local"])

    assert res.exit_code != 0
    assert "--keep_local" in res.output


def test_an_explicit_scan_dir_opts_back_into_shared_storage(genmodels, batch_env, tmp_path):
    res = CliRunner().invoke(
        genmodels.main, ["--scan_dir", str(tmp_path / "shared"), "--num_jobs", "300",
                         "--runid", "90009", "--backend", "htcondor", "--publish"])

    assert res.exit_code == 0, res.output
    scan_parent = _config_value(batch_env["argv"], "scan_parent")
    assert scan_parent == str(tmp_path / "shared")
    # one root again: nothing is ephemeral, so the split serves no purpose
    assert _config_value(batch_env["argv"], "meta_parent") == scan_parent


def test_an_explicit_scan_dir_warns_about_the_footprint(genmodels, batch_env,
                                                        tmp_path, monkeypatch):
    '''The failure mode is silent until the volume fills, so the warning has to
    quantify the peak for THIS run rather than just note that files are large.'''
    warnings = []
    monkeypatch.setattr(genmodels.log, "warning", lambda msg, *a, **k: warnings.append(str(msg)))

    res = CliRunner().invoke(
        genmodels.main, ["--scan_dir", str(tmp_path / "shared"), "--num_jobs", "300",
                         "--runid", "90009", "--backend", "htcondor", "--publish"])

    assert res.exit_code == 0, res.output
    assert any("300 GB" in w for w in warnings), warnings
    assert any("_CONDOR_SCRATCH_DIR" in w for w in warnings), warnings


@pytest.mark.parametrize("backend", ["local", "slurm"])
def test_other_backends_keep_the_shared_fs_layout(genmodels, batch_env, backend, tmp_path):
    '''Only HTCondor hands each job a private scratch dir it then deletes; the slurm
    flow writes to a shared /ptmp area and the local flow has no job at all.'''
    res = CliRunner().invoke(genmodels.main, _BATCH + ["--backend", backend])

    assert res.exit_code == 0, res.output
    scan_parent = _config_value(batch_env["argv"], "scan_parent")
    assert scan_parent == str(tmp_path / "scan")
    assert _config_value(batch_env["argv"], "meta_parent") == scan_parent


def test_single_mode_is_unaffected_on_htcondor(genmodels, batch_env, tmp_path):
    '''One scan, one job: the ntuple is the deliverable and it has to survive, so
    single mode keeps writing to shared storage even on htcondor.'''
    res = CliRunner().invoke(genmodels.main, ["--scan_dir", "relscan", "--backend", "htcondor"])

    assert res.exit_code == 0, res.output
    assert _config_value(batch_env["argv"], "scan_dir") == str(tmp_path / "relscan")


# ── rerunStep.py ───────────────────────────────────────────────────────────

@pytest.fixture
def existing_scan(tmp_path, default_config):
    '''A scan dir shaped like one the workflow produced, for rerunStep to resume.

    rerunStep recovers the step list from the merged config (the recorded
    scan_config.yaml names no config_file, so that is the default one) and
    refuses to start unless every preceding step has a manifest — so give it a
    complete set.
    '''
    from Run3ModelGen import pipeline

    scan = tmp_path / "relscan"
    scan.mkdir()
    (scan / "scan_config.yaml").write_text(yaml.safe_dump(
        {"prior": "flat", "seed": 123, "config_file": None}))
    (scan / "ntuple.0.0.root").write_bytes(b"root")

    meta = tmp_path / "r3mg" / "relscan"
    meta.mkdir(parents=True)
    (meta / "points.npz").write_bytes(b"")     # rerunStep refuses a scan without it
    cfg = pipeline.load_scan_config(None, default_config)
    for step in pipeline.resolve_steps(cfg["steps"]):
        (meta / f"step_{step['name']}.json").write_text("{}")
    return scan


@pytest.mark.parametrize("backend", ["local", "htcondor"])
def test_rerunstep_passes_an_absolute_scan_dir(rerunstep, captured_argv, existing_scan,
                                               tmp_path, monkeypatch, backend):
    seen = captured_argv(rerunstep)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rerunstep.snakefacade, "preflight", lambda *a, **k: [])

    res = CliRunner().invoke(rerunstep.main,
                             ["--step", "gm2calc", "--scan_dir", "relscan",
                              "--backend", backend])

    assert res.exit_code == 0, res.output
    scan_dir = _config_value(seen["argv"], "scan_dir")
    assert os.path.isabs(scan_dir), f"relative scan_dir would resolve on the worker: {scan_dir}"
    assert scan_dir == str(existing_scan)
