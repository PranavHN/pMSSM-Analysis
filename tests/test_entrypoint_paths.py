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
