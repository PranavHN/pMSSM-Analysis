'''Backend plumbing: snakemake argv, scheduler flags, pre-flight and resume guards.

These assertions are the only automated check on the HTCondor mapping — no
condor scheduler is reachable from the development or CI machines, so the
JobFlavour quoting is verified here rather than by running a job.
'''

import os

import pytest
import yaml

from Run3ModelGen import snakefacade as sf


@pytest.fixture(autouse=True)
def _repo_env(monkeypatch, repo):
    monkeypatch.setenv("RUN3MODELGENDIR", str(repo))


# ── argv construction ──────────────────────────────────────────────────────

def test_argv_points_at_the_workflow_and_the_requested_profile(repo):
    argv = sf.snakemake_argv("slurm", {"mode": "batch"})
    assert argv[0] == "snakemake"
    assert argv[argv.index("-s") + 1] == f"{repo}/workflow/Snakefile"
    assert argv[argv.index("--workflow-profile") + 1] == f"{repo}/workflow/profiles/slurm"


def test_argv_serialises_config_and_skips_none_values():
    argv = sf.snakemake_argv("local", {"mode": "single", "seed": 123, "config_file": None})
    tail = argv[argv.index("--config") + 1:]
    assert tail == ["mode=single", "seed=123"], "None entries must not reach snakemake"


def test_argv_places_targets_after_a_double_dash():
    argv = sf.snakemake_argv("local", {"mode": "extract"}, targets=["extract_all"])
    assert argv[-2:] == ["--", "extract_all"]


def test_argv_keeps_extra_flags_before_config():
    argv = sf.snakemake_argv("local", {"mode": "single"}, extra=["--rerun-triggers", "mtime"])
    assert argv.index("--rerun-triggers") < argv.index("--config")


# ── per-backend scheduler flags ────────────────────────────────────────────

def test_local_backend_adds_no_scheduler_flags():
    assert sf.scheduler_extra("local") == []
    assert sf.scheduler_extra("local", log_dir="/x") == []


def test_slurm_keeps_successful_job_logs_in_the_run_log_dir():
    '''Without these the plugin deletes the logs of jobs that succeeded.'''
    extra = sf.scheduler_extra("slurm", log_dir="/runs/log")
    assert extra == ["--slurm-logdir", "/runs/log/outfiles", "--slurm-keep-successful-logs"]


def test_htcondor_passes_jobdir_and_a_complete_resource_set():
    extra = sf.scheduler_extra("htcondor", log_dir="/runs/log", flavour="espresso")
    assert extra[:2] == ["--htcondor-jobdir", "/runs/log/outfiles"]
    assert extra[2] == "--default-resources"
    resources = extra[3:]
    # a CLI --default-resources replaces the profile's set, so every value the
    # profile provides has to be present here as well
    assert any(r.startswith("classad_MaxRuntime=") for r in resources)
    assert any(r.startswith("htcondor_request_disk_mb=") for r in resources)
    assert any(r.startswith("htcondor_request_mem_mb=") for r in resources)


def test_jobflavour_is_a_quoted_expression_around_the_bare_flavour():
    '''snakemake needs quotes to keep the value a string; the htcondor plugin
    then adds the quotes the submit file needs (+JobFlavour = "espresso").
    Embedding them here instead produces ""espresso"" and condor refuses the
    submit description ("Parse error in expression").'''
    entry = next(r for r in sf.flavour_resources("espresso") if r.startswith("classad_JobFlavour="))
    assert entry == "classad_JobFlavour='espresso'"


def test_htcondor_defaults_to_the_tomorrow_flavour():
    entry = next(r for r in sf.scheduler_extra("htcondor", "/l") if "JobFlavour" in r)
    assert entry == "classad_JobFlavour='tomorrow'"


# ── pre-flight ─────────────────────────────────────────────────────────────

def test_preflight_is_quiet_for_a_plain_local_run():
    assert sf.preflight("local") == []


def test_preflight_reports_a_missing_scheduler_binary(monkeypatch):
    monkeypatch.setattr(sf.shutil, "which", lambda exe: None)
    problems = sf.preflight("slurm")
    assert problems and "sbatch" in problems[0]


def test_preflight_accepts_a_present_scheduler(monkeypatch):
    monkeypatch.setattr(sf.shutil, "which", lambda exe: "/usr/bin/" + exe)
    assert sf.preflight("htcondor") == []


def test_publish_requires_eospath(monkeypatch):
    monkeypatch.delenv("EOSPATH", raising=False)
    problems = sf.preflight("local", publish=True)
    assert problems and "EOSPATH" in problems[0]


def test_publish_accepts_a_writable_destination(monkeypatch, tmp_path):
    monkeypatch.setenv("EOSPATH", str(tmp_path / "eos"))
    assert sf.preflight("local", publish=True) == [], "a creatable path is fine (publish mkdirs it)"


def test_publish_accepts_an_xrootd_url(monkeypatch):
    monkeypatch.setenv("EOSPATH", "root://eosuser.cern.ch//eos/user/x/y")
    assert sf.preflight("local", publish=True) == []


def test_publish_rejects_an_unwritable_destination(monkeypatch, tmp_path):
    '''Permission-independent on purpose: CI runs as root, where
    os.access(..., W_OK) is true even for /proc, so the filesystem cannot
    express "unwritable" portably. The check itself is what matters here.
    '''
    monkeypatch.setenv("EOSPATH", str(tmp_path / "eos"))
    monkeypatch.setattr(sf.os, "access", lambda path, mode: False)
    problems = sf.preflight("local", publish=True)
    assert problems and "not writable" in problems[0]


def test_publish_rejects_a_destination_blocked_by_a_file(monkeypatch, tmp_path):
    '''mkdir -p cannot create a directory below a regular file, so an existing
    non-directory ancestor must be reported instead of skipped over.'''
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setenv("EOSPATH", str(blocker / "runs"))
    problems = sf.preflight("local", publish=True)
    assert problems and "not a directory" in problems[0]


# ── lock detection ─────────────────────────────────────────────────────────

def test_locks_active_is_false_without_a_lock_dir(tmp_path):
    assert sf.locks_active(str(tmp_path)) is False


def test_locks_active_is_true_when_a_lock_is_present(tmp_path):
    locks = tmp_path / ".snakemake" / "locks"
    locks.mkdir(parents=True)
    (locks / "1.input.lock").touch()
    assert sf.locks_active(str(tmp_path)) is True


# ── run metadata: the resume guard that replaced --force ───────────────────

def _meta(**over):
    base = {"config_sha256": "abc", "base_seed": 123, "num_jobs": 2,
            "publish": False, "backend": "local", "orchestrator_pid": None}
    base.update(over)
    return base


def test_identical_resubmission_is_a_clean_resume():
    fatal, warn = sf.check_run_meta(_meta(), _meta())
    assert not fatal and not warn


def test_changed_config_content_is_refused():
    fatal, _ = sf.check_run_meta(_meta(config_sha256="old"), _meta(config_sha256="new"))
    assert fatal and "re-sample" in fatal[0]


def test_changed_base_seed_is_refused():
    fatal, _ = sf.check_run_meta(_meta(base_seed=1), _meta(base_seed=2))
    assert fatal


def test_growing_num_jobs_warns_but_proceeds():
    fatal, warn = sf.check_run_meta(_meta(num_jobs=2), _meta(num_jobs=5))
    assert not fatal and warn and "2 to 5" in warn[0]


def test_adding_publish_later_is_allowed():
    fatal, warn = sf.check_run_meta(_meta(publish=False), _meta(publish=True))
    assert not fatal and not warn


def test_a_dead_orchestrator_pid_does_not_block():
    fatal, _ = sf.check_run_meta(_meta(orchestrator_pid=2 ** 22 - 1), _meta())
    assert not fatal


def test_a_live_snakemake_orchestrator_blocks(monkeypatch):
    monkeypatch.setattr(sf, "_pid_running", lambda pid: pid is not None)
    fatal, _ = sf.check_run_meta(_meta(orchestrator_pid=4242), _meta())
    assert fatal and "4242" in fatal[0]


def test_a_live_but_unrelated_pid_does_not_block():
    '''pid recycling: only processes whose cmdline mentions snakemake count.'''
    assert sf._pid_running(os.getpid()) is False


def test_run_meta_roundtrip(tmp_path):
    path = sf.write_run_meta(str(tmp_path), "250727", _meta(num_jobs=4))
    assert path == sf.run_meta_path(str(tmp_path), "250727")
    with open(path) as f:
        assert yaml.safe_load(f)["num_jobs"] == 4


# ── config digest ──────────────────────────────────────────────────────────

def test_config_digest_tracks_content_not_filename(tmp_path, default_config):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("num_models: 5\n")
    b.write_text("num_models: 5\n")
    assert sf.config_digest(str(a), default_config) == sf.config_digest(str(b), default_config)
    b.write_text("num_models: 6\n")
    assert sf.config_digest(str(a), default_config) != sf.config_digest(str(b), default_config)


def test_config_digest_falls_back_to_the_default_config(default_config):
    assert sf.config_digest(None, default_config) == sf.config_digest(default_config, default_config)


# ── run id ─────────────────────────────────────────────────────────────────

def test_default_runid_is_digits_only():
    '''The scan.<C>.<P> naming contract cannot take dots or letters.'''
    assert sf.default_runid().isdigit()


# ── will a detached orchestrator survive logout? ───────────────────────────

class _Loginctl:
    '''Stand-in for `loginctl show-user`, which is absent off systemd hosts.'''
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode = stdout, returncode

    def __call__(self, *args, **kwargs):
        return self


def test_linger_off_is_reported(monkeypatch):
    '''The failure that killed two detached orchestrators mid-run.'''
    monkeypatch.setattr(sf.subprocess, "run", _Loginctl("State=active\nLinger=no\n"))
    warnings = sf.linger_warnings()
    assert len(warnings) == 1
    assert "enable-linger" in warnings[0], "must name the fix, not just the problem"
    # the consequence users actually have to act on: no automatic retry
    assert "RETRIES" in warnings[0].upper() and "re-run" in warnings[0]


def test_linger_on_is_silent(monkeypatch):
    monkeypatch.setattr(sf.subprocess, "run", _Loginctl("State=active\nLinger=yes\n"))
    assert sf.linger_warnings() == []


def test_linger_substring_is_not_mistaken_for_the_setting(monkeypatch):
    '''Matching "Linger=no" loosely would fire on e.g. LingerSomething=nope.'''
    monkeypatch.setattr(sf.subprocess, "run", _Loginctl("LingerFoo=nope\nLinger=yes\n"))
    assert sf.linger_warnings() == []


def test_no_loginctl_means_no_opinion(monkeypatch):
    '''macOS/containers have no loginctl — a guess there would be noise.'''
    def _missing(*args, **kwargs):
        raise FileNotFoundError("loginctl")
    monkeypatch.setattr(sf.subprocess, "run", _missing)
    assert sf.linger_warnings() == []


def test_loginctl_failure_means_no_opinion(monkeypatch):
    monkeypatch.setattr(sf.subprocess, "run", _Loginctl("", returncode=1))
    assert sf.linger_warnings() == []


def test_run_detached_forwards_the_warnings(monkeypatch, tmp_path):
    '''Every entrypoint that detaches must surface this, so it lives in
    run_detached rather than being repeated at four call sites.'''
    monkeypatch.setattr(sf, "linger_warnings", lambda: ["watch out"])
    monkeypatch.setattr(sf.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    seen = []
    pid = sf.run_detached(["true"], str(tmp_path / "log" / "snakemake.log"), warn=seen.append)
    assert pid == 4242 and seen == ["watch out"]


def test_run_detached_without_a_warn_callback_is_quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(sf, "linger_warnings", lambda: ["ignored"])
    monkeypatch.setattr(sf.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 7})())
    assert sf.run_detached(["true"], str(tmp_path / "snakemake.log")) == 7


# ── --default-resources REPLACES the profile block, so they must agree ──────

def test_flavour_resources_reproduces_the_profile_exactly(repo):
    '''snakemake's --default-resources replaces the profile's block instead of
    merging. The entrypoints always pass it, so any key declared only in
    workflow/profiles/htcondor/config.yaml is silently dropped for every real
    run — which is exactly how max_retries went missing.

    Compare VALUES, not just key names. An earlier version of this test compared
    key sets only, so editing a value in the profile alone (mem 4000 -> 3000) kept
    the suite green while every submission carried on using the facade's copy. The
    facade now derives its values from the profile, so this pins that it really
    round-trips rather than merely covering the same keys.'''
    import yaml

    profile = yaml.safe_load(
        (repo / "workflow" / "profiles" / "htcondor" / "config.yaml").read_text())
    declared = dict(profile.get("default-resources", {}))
    # classad_JobFlavour is the one per-run value the facade substitutes.
    expected = {key: str(value) for key, value in declared.items()}
    expected["classad_JobFlavour"] = "'espresso'"

    got = dict(entry.split("=", 1) for entry in sf.flavour_resources("espresso"))
    assert got == expected, (
        "flavour_resources() and the htcondor profile's default-resources disagree; "
        "the CLI copy wins at submit time, so the profile would be a silent lie")



# ── dropping --publish from a published run ────────────────────────────────

def _meta(**kw):
    base = {"config_sha256": "abc", "base_seed": 123, "num_jobs": 2,
            "publish": True, "keep_local": False, "orchestrator_pid": None}
    base.update(kw)
    return base


def test_dropping_publish_after_a_move_is_refused(repo):
    '''The locals are gone, so the ntuples cannot be targets — and the failure it
    would otherwise hit advises --forceall, i.e. regenerating complete batches.'''
    fatal, _ = sf.check_run_meta(_meta(), _meta(publish=False))
    assert any("--publish to resume" in f for f in fatal), fatal


def test_dropping_publish_is_fine_when_the_locals_were_kept(repo):
    '''--keep_local left the scan dirs in place, so the ntuples still exist.'''
    fatal, _ = sf.check_run_meta(_meta(keep_local=True), _meta(keep_local=True, publish=False))
    assert fatal == []


def test_resuming_a_published_run_with_publish_is_fine(repo):
    fatal, _ = sf.check_run_meta(_meta(), _meta())
    assert fatal == []


def test_adding_publish_later_is_still_allowed(repo):
    '''Documented behaviour: an unpublished run can be published afterwards.'''
    fatal, _ = sf.check_run_meta(_meta(publish=False), _meta(publish=True))
    assert fatal == []
