'''Guards on .gitlab-ci.yml for failure modes that only show up on the runner.

Each test here corresponds to a pipeline failure that reached CI: a job config
GitLab rejects, a SIGPIPE from `tar | grep -q` under pipefail, and a tool that
does not exist in the CI image. They are cheap and local, so the next instance
of these classes fails in a second instead of after a ten-minute scan.
'''

import os

import yaml
import pytest


class _Loader(yaml.SafeLoader):
    pass


# GitLab tags (!reference) are valid entries — represent them as strings so they
# are not mistaken for the mis-parsed mappings these tests hunt for.
_Loader.add_multi_constructor('!', lambda loader, suffix, node: '<gitlab-tag>')

SCRIPT_KEYS = ("script", "before_script", "after_script")


@pytest.fixture(scope="module")
def ci(repo):
    with open(repo / ".gitlab-ci.yml") as f:
        return yaml.load(f, Loader=_Loader)


def _script_entries(ci):
    '''Yield (job, key, index, entry) for every script entry in the file.'''
    for job, body in ci.items():
        if not isinstance(body, dict):
            continue
        for key in SCRIPT_KEYS:
            entries = body.get(key)
            if entries is None:
                continue
            if isinstance(entries, str):
                yield job, key, 0, entries
                continue
            for i, entry in enumerate(entries):
                yield job, key, i, entry


def test_every_script_entry_is_a_string(ci):
    '''GitLab: "script config should be a string or a nested array of strings".

    An unquoted entry containing ': ' silently parses as a mapping — the file is
    valid YAML, but the job is rejected. Plain parsing does not catch it.
    '''
    bad = []
    for job, key, i, entry in _script_entries(ci):
        if isinstance(entry, str):
            continue
        if isinstance(entry, list) and all(isinstance(x, str) for x in entry):
            continue
        bad.append(f"{job}.{key}[{i}] parsed as {type(entry).__name__}: {str(entry)[:120]}")
    assert not bad, ("non-string script entries (quote them, or drop the ': ' that makes "
                     "YAML read them as a mapping):\n" + "\n".join(bad))


def test_no_pipe_into_grep_q(ci):
    '''`producer | grep -q x` exits 141: grep -q closes the pipe, the producer
    gets SIGPIPE, and the runner's pipefail turns that into a job failure.
    Write to a file first, then grep the file.'''
    offenders = [f"{job}.{key}[{i}]: {entry}"
                 for job, key, i, entry in _script_entries(ci)
                 if isinstance(entry, str) and "| grep -q" in entry]
    assert not offenders, "SIGPIPE-prone pipelines:\n" + "\n".join(offenders)


def test_no_tools_missing_from_the_ci_image(ci):
    '''diffutils is in neither the alma9 base image nor the pixi environment,
    so cmp/diff exit 127 there. Compare in python instead.'''
    missing = ("cmp", "diff")
    offenders = []
    for job, key, i, entry in _script_entries(ci):
        if not isinstance(entry, str):
            continue
        first = entry.strip().lstrip("!").strip().split()
        if first and first[0] in missing:
            offenders.append(f"{job}.{key}[{i}]: {entry}")
    assert not offenders, ("tools absent from the CI image:\n" + "\n".join(offenders))


# compare_reference-gated priors — every one of them, mcmc included. mcmc is gated on
# branch VALUES as well as counts (compare_values in refcheck.SCANS), because an emcee
# run at a fixed seed reproduces exactly (tests/test_emcee_determinism.py) while its
# counts are near-vacuous: every model clears every step.
ALL_PRIORS = ("flat", "fixed", "mcmc", "grid", "importance")
# The docker track exists to validate the published image, not to re-prove the
# physics: both tracks resolve from the same pixi.lock, so a representative
# subset is enough there. flat covers the random priors; mcmc is included because the
# chain exercises the image's emcee/h5py/arviz stack, which no other job touches.
DOCKER_PRIORS = ("flat", "mcmc")


@pytest.mark.parametrize("scan", ALL_PRIORS)
def test_every_prior_is_regression_tested_locally(ci, scan):
    '''The deterministic local scans are the physics regression harness: each must
    run its scan and then compare_reference.py. Losing one silently would hide drift.'''
    job = f"genModels_{scan}_local"
    assert job in ci, f"{job} disappeared from CI"
    script = " ".join(ci[job]["script"])
    assert "genModels.py" in script
    assert f"compare_reference.py --scan {scan}" in script


@pytest.mark.parametrize("scan", DOCKER_PRIORS)
def test_the_docker_track_validates_the_image_on_a_prior_subset(ci, scan):
    job = f"genModels_{scan}_docker"
    assert job in ci, f"{job} disappeared from CI"
    assert f"compare_reference.py --scan {scan}" in " ".join(ci[job]["script"])


@pytest.mark.parametrize("track", ["local", "docker"])
def test_mcmc_is_value_gated_not_just_smoke_tested(ci, track):
    '''prior: mcmc must do a real reference comparison, not only assert that files
    appeared. It keeps the chain-specific smoke checks (the checkpoint and the reserved
    likelihood branches, which counts/values cannot express) AND runs
    compare_reference.py, which for this scan compares branch values too.'''
    job = f"genModels_mcmc_{track}"
    assert job in ci, f"{job} disappeared from CI"
    script = " ".join(ci[job]["script"])
    assert "genModels.py" in script and "ci_mcmc_config.yaml" in script
    assert "compare_reference.py --scan mcmc" in script, \
        "mcmc must be reference-compared, not merely smoke-tested"
    assert "chain.h5" in script, "the mcmc job must assert the emcee checkpoint was produced"
    assert "emcee_logp" in script, "the reserved likelihood branches must still be checked"


def test_the_value_gated_scan_relies_on_the_default_seed(ci, repo):
    '''A value gate is only meaningful if the reference and the CI run use the same seed.
    Neither the CI job nor make_references.py passes --seed, so both take genModels'
    default — which therefore must not move without regenerating the references.'''
    import re

    src = (repo / "source" / "Run3ModelGen" / "scripts" / "genModels.py").read_text()
    m = re.search(r"'--seed',\s*default=(\d+)", src)
    assert m, "could not find genModels' --seed default"
    assert m.group(1) == "123", (
        "genModels' default seed changed; the mcmc value reference was built with 123, "
        "so regenerate it (make-refs + upload-refs) or this gate fails for everyone")

    for track in ("local", "docker"):
        script = " ".join(ci[f"genModels_mcmc_{track}"]["script"])
        assert "--seed" not in script, (
            "the mcmc CI job must not pin --seed: make_references.py does not either, "
            "and the two must agree")

    refs = (repo / "source" / "Run3ModelGen" / "scripts" / "make_references.py").read_text()
    assert "--seed" not in refs, "make_references.py must not pin --seed either"


def test_mcmc_is_the_scan_that_compares_values(ci):
    '''The value comparison is what makes gating the chain worthwhile; a plain count
    gate on it would pass on 54/54 regardless of where the walkers went.'''
    from Run3ModelGen.refcheck import by_name, SCANS
    assert by_name("mcmc").get("compare_values") is True
    others = [s["name"] for s in SCANS if s["name"] != "mcmc" and s.get("compare_values")]
    assert not others, (f"{others} also compare values — fine in principle, but they "
                        f"sample random points, so update this test deliberately")


def test_publish_waits_for_every_test_job_that_exists(ci):
    '''A `needs:` entry naming a deleted job is a hard pipeline error, and a
    missing one would publish an image whose tests never ran.'''
    jobs = {k for k, v in ci.items() if isinstance(v, dict) and "stage" in v}
    required = {n["job"] if isinstance(n, dict) else n for n in ci["publish_docker"]["needs"]}
    assert required <= jobs, f"publish_docker needs jobs that do not exist: {required - jobs}"
    tested = {j for j in jobs if ci[j].get("stage") in ("genModels", "extractModels")}
    # every docker-track test job must gate the publish
    docker_tests = {j for j in tested if j.endswith("_docker")}
    assert docker_tests <= required, f"publish_docker does not wait for {docker_tests - required}"


def test_batch_job_asserts_the_published_layout_and_extraction(ci):
    '''Batch mode's contracts (publish layout, model-id offsets, extraction)
    have no other automated coverage.'''
    script = " ".join(ci["genModels_batch_local"]["script"])
    assert "--num_jobs 2" in script and "--publish" in script
    assert "fake_eos_pub/90001/1/scan.90001.1.tar.gz" in script
    assert "100000, 100001, 100002" in script
    assert "--eos_dir" in script
    assert "rerunStep.py" in script


def test_batch_job_covers_publish_as_a_move(ci):
    '''Default --publish deletes each batch's local scan dir + tarball. Both
    halves need coverage: that the local copies really go, and that a re-run is
    then still a no-op (a regression there silently re-runs whole scans).'''
    script = " ".join(ci["genModels_batch_local"]["script"])
    assert "! test -e movetest/scan.90002.1" in script, "no check that the scan dir is gone"
    assert "! test -e movetest/scan.90002.1.tar.gz" in script, "no check that the tarball is gone"
    assert "movetest/r3mg/scan.90002.1/published.ok" in script, "bookkeeping must survive"
    assert "Nothing to be done" in script, "no resume check after a publishing move"
    # the keep_local escape hatch must stay exercised too, or the assertions
    # that read local files would start passing for the wrong reason
    assert "--keep_local" in script


@pytest.fixture(scope="module")
def checks_script(repo):
    return (repo / "tests" / "run_checks.sh").read_text()


def test_rule_bodies_referenced_by_the_workflow_exist(repo):
    '''Rule bodies are invoked by path now, and a dry-run does NOT check that a
    rule's command exists — a typo would only surface mid-run on a worker. Cheap
    guard: every {RULE_SCRIPTS}/<file> the rules mention must be an executable file.'''
    import re

    rule_scripts = repo / "workflow" / "scripts"
    referenced = set()
    for smk in (repo / "workflow").rglob("*.smk"):
        referenced |= set(re.findall(r"\{RULE_SCRIPTS\}/([\w.\-]+)", smk.read_text()))
    assert referenced, "no rule bodies referenced — did the placeholder name change?"
    for name in sorted(referenced):
        path = rule_scripts / name
        assert path.is_file(), f"workflow rules invoke {name}, which does not exist in workflow/scripts/"
        assert os.access(path, os.X_OK), f"{name} is not executable"


def test_offline_checks_have_a_single_definition(ci, repo, checks_script):
    '''`pixi run test` and the CI dry-run job must run the SAME checks: both
    call run_checks.sh, so neither can quietly fall behind the other.'''
    import tomllib

    assert "run_checks.sh" in " ".join(ci["workflow_dryrun_local"]["script"])
    with open(repo / "pixi.toml", "rb") as f:
        assert "run_checks.sh" in str(tomllib.load(f)["tasks"]["test"])


def test_offline_checks_cover_tests_and_every_workflow_mode(checks_script):
    '''Each workflow mode must at least keep parsing and building its DAG.'''
    assert "pytest tests" in checks_script
    for mode in ("mode=single", "mode=batch", "mode=extract"):
        assert mode in checks_script
    assert "publish=eos" in checks_script
    assert "ci_mcmc_config.yaml" in checks_script, "the mcmc chain DAG needs coverage too"


def test_the_sampled_point_golden_file_is_guarded(repo):
    '''The RNG contract underpins the per-step reference counts: the golden file
    and the test that checks it must both exist, with a way to regenerate it.'''
    assert (repo / "tests" / "fixtures" / "sampling_seed123.json").is_file()
    assert (repo / "tests" / "test_sampling_fixture.py").is_file()
    assert (repo / "tests" / "regen_sampling_fixture.py").is_file()


def test_offline_checks_fail_fast(checks_script):
    '''Without set -e a failing check would be masked by the next one.'''
    assert "set -euo pipefail" in checks_script


def test_the_cheap_job_gates_the_expensive_scans(ci):
    '''A failing contract check must stop the pipeline before ten multi-minute
    scans run, and the gate job itself must not depend on anything but the build
    (in particular not on the kinit-carrying template).'''
    gate = "workflow_dryrun_local"
    assert ci[gate]["extends"] == ".setup_local", (
        "the gate needs no EOS access; extending .genmodels_local would make it "
        "depend on the masked kinit credentials")
    assert ci[gate]["needs"] == ["build_local"]
    scans = [j for j in ci if j.startswith("genModels_") and j != gate]
    assert scans
    for job in scans:
        assert gate in ci[job]["needs"], f"{job} does not wait for {gate}"


def test_smodels_step_is_serialised_against_concurrent_batches(repo):
    '''SModelS holds the ~1.4 GB results database in memory per model. Two
    batches doing that at once got the process OOM-killed in CI (a bare
    non-zero exit, no traceback). The rule must claim a token and the local
    profile must cap it at one.'''
    import yaml as _yaml

    rules = (repo / "workflow" / "rules" / "generate.smk").read_text()
    assert "smodels_db" in rules, "rule step no longer claims the smodels_db token"
    assert 'if w.step == "SModelS"' in rules, "the token must apply to the SModelS step only"

    with open(repo / "workflow" / "profiles" / "local" / "config.yaml") as f:
        profile = _yaml.safe_load(f)
    assert "smodels_db=1" in [str(r) for r in profile.get("resources", [])], \
        "the local profile must limit smodels_db to 1, otherwise the token is unbounded"


# ── pooled cross-ensemble diagnostics ──────────────────────────────────────

def test_pooled_diag_rule_exists_and_stays_out_of_the_default_target(repo):
    '''chain_diag_all pools the per-seed backends for the AUTHORITATIVE numbers (cross-
    ensemble R-hat, the cross-seed rank statistic, octant spread), because each chain job
    only ever sees its own chain.h5 and its diag/ is therefore single-ensemble.

    Two properties matter and neither is obvious from reading the rule alone:
      * it must NOT be a default target -- an arviz/plotting hiccup must not fail an
        otherwise-complete production run, the same contract the per-batch diagnostics
        already have (they are best-effort inside r3mg_batch.py);
      * it must take the NTUPLES as input, never chain.h5, which is undeclared
        side-state with no producing rule -- listing it would fail at DAG build.
    '''
    smk = (repo / "workflow" / "rules" / "chain.smk").read_text()
    assert "rule chain_diag_all:" in smk
    body = smk.split("rule chain_diag_all:")[1]
    assert "ntuple" in body.split("output:")[0], "must depend on the ntuples"
    assert "chain.h5" not in body.split("output:")[0], \
        "must NOT declare chain.h5 as an input (undeclared side-state)"
    assert "emcee_diagnostics.py" in body

    snakefile = (repo / "workflow" / "Snakefile").read_text()
    targets = snakefile.split("def _final_targets")[1].split("rule all")[0]
    assert "diag_all" not in targets and "chain_diag_all" not in targets, \
        "chain_diag_all must not be wired into the default target"


def test_pooled_diag_refuses_a_single_backend(repo):
    '''Under htcondor scratch scan dirs, or --publish without --keep_local, the backends
    are scattered across workers or already gone. The glob would then match one file and
    silently produce a "cross-ensemble" report built from a single chain -- precisely the
    misreading this rule exists to prevent -- so it must fail instead.'''
    smk = (repo / "workflow" / "rules" / "chain.smk").read_text()
    body = smk.split("rule chain_diag_all:")[1]
    assert "-lt 2" in body, "must require at least two backends"
    assert "exit 1" in body, "must fail rather than diagnose one chain as if pooled"
