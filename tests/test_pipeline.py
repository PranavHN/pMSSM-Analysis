'''Config merge, step resolution, batch identity and manifest contracts.'''

import json

import pytest
import yaml

from Run3ModelGen import pipeline


# ── config merge ───────────────────────────────────────────────────────────

def test_default_config_is_used_when_no_user_config(default_config):
    cfg = pipeline.load_scan_config(None, default_config)
    assert cfg["prior"] == "flat"
    assert cfg["num_models"] == 20
    assert [s["name"] for s in cfg["steps"]][0] == "prep_input"


def test_user_keys_override_defaults(tmp_path, default_config):
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({"num_models": 7, "prior": "flat"}))
    cfg = pipeline.load_scan_config(str(user), default_config)
    assert cfg["num_models"] == 7
    # untouched keys still come from the default
    assert "steps" in cfg and cfg["isGMSB"] is False


def test_extra_user_keys_are_carried_through(tmp_path, default_config):
    '''grid/importance blocks are not in the default config and must survive.'''
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({"prior": "grid", "grid": {"M_1": [1, 2, 3]}}))
    cfg = pipeline.load_scan_config(str(user), default_config)
    assert cfg["grid"] == {"M_1": [1, 2, 3]}


def test_fixed_prior_infers_num_models_from_parameter_lists(tmp_path, default_config):
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({
        "prior": "fixed",
        "parameters": {"tanb": [10, 20, 30, 40]},
    }))
    cfg = pipeline.load_scan_config(str(user), default_config)
    assert cfg["num_models"] == 4


def test_explicit_num_models_wins_over_fixed_inference(tmp_path, default_config):
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({
        "prior": "fixed", "num_models": 2,
        "parameters": {"tanb": [10, 20, 30, 40]},
    }))
    assert pipeline.load_scan_config(str(user), default_config)["num_models"] == 2


# ── platform-conditional steps ─────────────────────────────────────────────

def test_resolve_steps_keeps_steps_without_an_executable_requirement():
    steps = [{"name": "prep_input"}, {"name": "SPheno"}, {"name": "SModelS"}]
    assert pipeline.resolve_steps(steps) == steps


def test_resolve_steps_drops_feynhiggs_when_the_binary_is_missing(monkeypatch, tmp_path):
    '''macOS has no conda-forge FeynHiggs: the step must vanish, not fail.

    Both oracles must come up empty — an env_bin of "" alone would still let a
    FeynHiggs on the real PATH keep the step (see tool_available).'''
    monkeypatch.setattr(pipeline.shutil, "which", lambda exe: None)
    steps = [{"name": "SPheno"}, {"name": "feynhiggs"}, {"name": "SModelS"}]
    resolved = pipeline.resolve_steps(steps, env_bin=str(tmp_path))
    assert [s["name"] for s in resolved] == ["SPheno", "SModelS"]


def test_resolve_steps_keeps_feynhiggs_when_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.shutil, "which", lambda exe: "/usr/bin/" + exe)
    steps = [{"name": "feynhiggs"}]
    resolved = pipeline.resolve_steps(steps, env_bin=str(tmp_path))
    assert [s["name"] for s in resolved] == ["feynhiggs"]


def test_resolve_steps_keeps_feynhiggs_when_only_in_the_pixi_env(monkeypatch, tmp_path):
    '''The parse-time case: PATH is not set up yet, the env still has the tool.'''
    monkeypatch.setattr(pipeline.shutil, "which", lambda exe: None)
    (tmp_path / "FeynHiggs").write_text("")
    resolved = pipeline.resolve_steps([{"name": "feynhiggs"}], env_bin=str(tmp_path))
    assert [s["name"] for s in resolved] == ["feynhiggs"]


def test_env_bin_dir_is_derived_from_the_build_environment(monkeypatch):
    monkeypatch.setenv("RUN3MODELGENDIR", "/some/checkout/")
    assert pipeline.env_bin_dir() == "/some/checkout/.pixi/envs/default/bin"


def test_env_bin_dir_is_empty_without_the_build_environment(monkeypatch):
    '''tool_available must then fall back to PATH alone, not stat "/.pixi/...".'''
    monkeypatch.delenv("RUN3MODELGENDIR", raising=False)
    assert pipeline.env_bin_dir() == ""


# ── per-job model cap ──────────────────────────────────────────────────────

def test_model_cap_allows_the_limit():
    pipeline.check_model_cap(pipeline.MAX_MODELS_PER_JOB)  # must not raise


def test_model_cap_rejects_more_than_the_limit():
    with pytest.raises(SystemExit):
        pipeline.check_model_cap(pipeline.MAX_MODELS_PER_JOB + 1)


# ── batch identity ─────────────────────────────────────────────────────────

def test_batch_seed_is_deterministic():
    assert pipeline.batch_seed("250727", 123, 3) == pipeline.batch_seed("250727", 123, 3)


def test_batch_seed_differs_per_batch_runid_and_base_seed():
    seeds = {
        pipeline.batch_seed("250727", 123, 0),
        pipeline.batch_seed("250727", 123, 1),
        pipeline.batch_seed("250728", 123, 0),
        pipeline.batch_seed("250727", 124, 0),
    }
    assert len(seeds) == 4


def test_batch_seed_fits_numpys_legacy_seed_bound():
    for b in range(50):
        seed = pipeline.batch_seed("1", 123, b)
        assert 0 <= seed < 2 ** 32


# ── naming contract (shared with ntupling.mkntuple / extractModels) ────────

@pytest.mark.parametrize("scan_dir, expected", [
    ("scan", "scan/ntuple.0.0.root"),
    ("run/scan_flat", "run/scan_flat/ntuple.0.0.root"),
    ("scan.90001.7", "scan.90001.7/ntuple.90001.7.root"),
    ("/ptmp/u/run/scan.250727.12", "/ptmp/u/run/scan.250727.12/ntuple.250727.12.root"),
    ("scan.90001.7/", "scan.90001.7/ntuple.90001.7.root"),
])
def test_ntuple_name_matches_the_scan_dir_convention(scan_dir, expected):
    assert pipeline.ntuple_name(scan_dir) == expected


# ── step manifests ─────────────────────────────────────────────────────────

def test_manifest_roundtrip_records_counts_and_gating(tmp_path):
    path = tmp_path / "meta" / "step_SPheno.json"
    pipeline.write_manifest(str(path), step="SPheno", num_models=20,
                            attempted=[0, 1, 2, 3], succeeded=[2, 0], chain_ok=[2, 0])
    m = pipeline.read_manifest(str(path))
    assert m["step"] == "SPheno"
    assert m["counts"] == [2, 20]
    assert m["attempted"] == [0, 1, 2, 3]
    assert m["succeeded"] == [0, 2], "entries must be sorted for stable diffs"
    assert m["chain_ok"] == [0, 2]


def test_manifest_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "step_x.json"
    pipeline.write_manifest(str(path), step="x", num_models=1,
                            attempted=[0], succeeded=[0], chain_ok=[0])
    assert path.exists()
    assert not (tmp_path / "step_x.json.tmp").exists()
    json.loads(path.read_text())  # valid JSON, fully written


def test_manifest_overwrite_replaces_previous_content(tmp_path):
    path = tmp_path / "step_x.json"
    pipeline.write_manifest(str(path), step="x", num_models=5,
                            attempted=[0, 1], succeeded=[0, 1], chain_ok=[0, 1])
    pipeline.write_manifest(str(path), step="x", num_models=5,
                            attempted=[0, 1], succeeded=[1], chain_ok=[1])
    assert pipeline.read_manifest(str(path))["counts"] == [1, 5]


# ── which batch job ran this step ──────────────────────────────────────────

@pytest.fixture
def clean_scheduler_env(monkeypatch):
    '''No scheduler variables, so each test sets exactly the ones it means.'''
    for var in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_RESTART_COUNT", "_CONDOR_JOB_AD"):
        monkeypatch.delenv(var, raising=False)


def test_scheduler_identity_reports_slurm(monkeypatch, clean_scheduler_env):
    monkeypatch.setenv("SLURM_JOB_ID", "4815162342")
    ident = pipeline.scheduler_identity()
    assert ident["backend"] == "slurm" and ident["job_id"] == "4815162342"
    assert ident["host"]


def test_scheduler_identity_accepts_the_older_slurm_spelling(monkeypatch, clean_scheduler_env):
    '''Older slurm exports SLURM_JOBID; the site version must not decide this.'''
    monkeypatch.setenv("SLURM_JOBID", "99")
    assert pipeline.scheduler_identity()["job_id"] == "99"


def test_scheduler_identity_records_a_slurm_requeue(monkeypatch, clean_scheduler_env):
    '''slurm-requeue keeps the job id, so the restart count is what tells the
    attempts apart.'''
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    assert pipeline.scheduler_identity()["restart_count"] == "2"


def test_scheduler_identity_reports_htcondor_from_the_job_ad(monkeypatch, clean_scheduler_env, tmp_path):
    ad = tmp_path / "job_ad"
    ad.write_text('ClusterId = 20423535\nProcId = 0\n'
                  'GlobalJobId = "bigbird25.cern.ch#20423535.0#1785238088"\nOwner = "jwuerzin"\n')
    monkeypatch.setenv("_CONDOR_JOB_AD", str(ad))
    ident = pipeline.scheduler_identity()
    assert ident["backend"] == "htcondor"
    assert ident["job_id"] == "20423535.0", "must be the form condor_q/condor_rm take"
    assert ident["global_job_id"].startswith("bigbird25.cern.ch#")


def test_scheduler_identity_falls_back_to_local(clean_scheduler_env):
    assert pipeline.scheduler_identity()["backend"] == "local"


def test_scheduler_identity_survives_an_unreadable_job_ad(monkeypatch, clean_scheduler_env, tmp_path):
    '''Metadata must never fail a rule: a bad ad path degrades, not raises.'''
    monkeypatch.setenv("_CONDOR_JOB_AD", str(tmp_path / "does_not_exist"))
    assert pipeline.scheduler_identity()["backend"] == "local"


def test_scheduler_identity_never_raises(monkeypatch, clean_scheduler_env):
    monkeypatch.setattr(pipeline.socket, "gethostname", lambda: 1 / 0)
    assert pipeline.scheduler_identity() == {}


def test_manifests_record_the_batch_job(tmp_path, monkeypatch, clean_scheduler_env):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    path = tmp_path / "step_SPheno.json"
    pipeline.write_manifest(str(path), step="SPheno", num_models=3,
                            attempted=[0, 1, 2], succeeded=[0, 2], chain_ok=[0, 2])
    m = pipeline.read_manifest(str(path))
    assert m["scheduler"]["job_id"] == "1234"
    # the fields every consumer reads must be untouched by the addition
    assert m["chain_ok"] == [0, 2] and m["counts"] == [2, 3] and m["step"] == "SPheno"
