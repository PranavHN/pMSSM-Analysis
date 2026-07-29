'''The Snakefile's parse-time mirrors must agree with pipeline.py.

Parse-time purity (see the Snakefile header) forbids the Snakefile from
importing Run3ModelGen: remote executors re-parse it inside every job, where
the package is not yet importable. Four pieces of logic are therefore
duplicated — the config merge, the platform step filter, the per-batch seed and
the ntuple naming contract — and a drift between the two copies is SILENT: the
DAG would describe a run that differs from what the rule bodies actually do
(a rule that never sees its own output, a step the body then refuses, a batch
sampled from a different seed than the one the DAG was built for).

Nothing else pins that agreement, so this module does. The mirrors are pulled
out of the Snakefile as source text — the file cannot be imported (`rule`,
`include:` and friends are not Python) — and compared against pipeline.py over
the repo's real configs plus edge cases.
'''

import hashlib
import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

from Run3ModelGen import pipeline

# Every mirrored helper. Missing one here means a rename went unnoticed.
MIRRORS = ("_load_cfg", "_resolve_steps", "_batch_seed", "_ntuple_basename")

# Collected at import time: parametrize needs the list before fixtures exist.
CONFIGS = sorted(p for p in (Path(__file__).resolve().parent.parent / "source" /
                             "Run3ModelGen" / "data" / "configs").glob("*.yaml")
                 if p.name != "default_config.yaml")


def _extract_mirrors(snakefile: str, names) -> str:
    '''Return the source text of the named top-level functions of a Snakefile.

    A Snakefile is not valid Python (so neither import nor ast.parse work on the
    whole file), but the mirrors themselves are ordinary defs: collect each
    header line plus its indented body.
    '''
    lines = snakefile.splitlines(keepends=True)
    blocks, found, i = [], set(), 0
    while i < len(lines):
        match = re.match(r"def (\w+)\(", lines[i])
        if match and match.group(1) in names:
            found.add(match.group(1))
            blocks.append(lines[i])
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
                blocks.append(lines[i])
                i += 1
        else:
            i += 1
    missing = set(names) - found
    assert not missing, (
        f"{sorted(missing)} no longer defined at the top level of the Snakefile. "
        f"If a mirror was renamed or removed, update MIRRORS — do not delete this "
        f"test: it is the only check that the two copies still agree.")
    return "".join(blocks)


@pytest.fixture(scope="session")
def snakefile_src(repo):
    return (repo / "workflow" / "Snakefile").read_text()


@pytest.fixture
def mirrors(snakefile_src, datadir):
    '''Factory: the Snakefile's mirror functions, bound to a given ENV_BIN.

    Mirrors the Snakefile's own module globals (DATA, ENV_BIN) — everything else
    they touch is stdlib.
    '''
    def _load(env_bin=""):
        ns = {"os": os, "hashlib": hashlib, "shutil": shutil, "yaml": yaml,
              "DATA": str(datadir), "ENV_BIN": env_bin}
        exec(_extract_mirrors(snakefile_src, MIRRORS), ns)
        return ns
    return _load


def test_every_mirror_is_still_present(mirrors):
    '''Guard the guard: extraction must find all four, or the rest is vacuous.'''
    ns = mirrors()
    for name in MIRRORS:
        assert callable(ns[name])


# ── config merge: _load_cfg vs pipeline.load_scan_config ───────────────────

def test_merge_agrees_with_no_user_config(mirrors, default_config):
    assert mirrors()["_load_cfg"]("") == pipeline.load_scan_config(None, default_config)


def test_the_shipped_configs_were_actually_found():
    '''A glob that silently matches nothing would make the sweep below vacuous.'''
    assert len(CONFIGS) >= 5


@pytest.mark.parametrize("config", [pytest.param(p, id=p.name) for p in CONFIGS])
def test_merge_agrees_on_every_shipped_config(mirrors, default_config, config):
    '''Every config the repo ships (and CI runs) must merge identically.'''
    assert mirrors()["_load_cfg"](str(config)) == \
        pipeline.load_scan_config(str(config), default_config)


@pytest.mark.parametrize("user", [
    pytest.param({}, id="empty"),
    pytest.param({"num_models": 7}, id="override-a-default-key"),
    pytest.param({"prior": "fixed", "parameters": {"tanb": [10, 20, 30]}},
                 id="fixed-infers-num_models"),
    pytest.param({"prior": "fixed", "num_models": 2,
                  "parameters": {"tanb": [10, 20, 30]}},
                 id="explicit-num_models-wins"),
    pytest.param({"prior": "grid", "grid": {"M_1": [1, 2, 3]}},
                 id="extra-key-not-in-default"),
    pytest.param({"isGMSB": None}, id="explicit-null-value"),
    pytest.param({"steps": []}, id="empty-step-list"),
])
def test_merge_agrees_on_edge_cases(mirrors, default_config, tmp_path, user):
    path = tmp_path / "u.yaml"
    path.write_text(yaml.safe_dump(user))
    assert mirrors()["_load_cfg"](str(path)) == \
        pipeline.load_scan_config(str(path), default_config)


# ── step filter: _resolve_steps vs pipeline.resolve_steps ──────────────────

STEPS = [{"name": "prep_input"}, {"name": "SPheno"},
         {"name": "feynhiggs"}, {"name": "SModelS"}]


@pytest.fixture
def tool_layout(tmp_path, monkeypatch):
    '''Factory placing FeynHiggs in the pixi env, on PATH, both or neither.

    These four combinations are the whole disagreement surface between the two
    oracles (env-by-absolute-path at parse time vs PATH at run time).
    '''
    def _make(in_env: bool, on_path: bool):
        env_bin = tmp_path / "envbin"
        other = tmp_path / "elsewhere"
        for d in (env_bin, other):
            d.mkdir(exist_ok=True)
        if in_env:
            # the env is checked by existence, but keep it a realistic binary
            (env_bin / "FeynHiggs").write_text("")
            (env_bin / "FeynHiggs").chmod(0o755)
        if on_path:
            # shutil.which only finds it if it is executable
            (other / "FeynHiggs").write_text("")
            (other / "FeynHiggs").chmod(0o755)
        monkeypatch.setenv("PATH", str(other))
        # never let the real checkout's env leak in through env_bin_dir()
        monkeypatch.delenv("RUN3MODELGENDIR", raising=False)
        return str(env_bin)
    return _make


@pytest.mark.parametrize("in_env,on_path", [(False, False), (True, False),
                                            (False, True), (True, True)])
def test_step_filter_agrees_wherever_the_tool_lives(mirrors, tool_layout,
                                                    in_env, on_path):
    env_bin = tool_layout(in_env, on_path)
    assert mirrors(env_bin)["_resolve_steps"](STEPS) == \
        pipeline.resolve_steps(STEPS, env_bin=env_bin)


@pytest.mark.parametrize("in_env,on_path,kept", [
    (False, False, False),   # macOS: no conda-forge build anywhere
    (True, False, True),     # the normal Linux layout
    (False, True, True),     # installed outside the env (module load, /usr/bin)
    (True, True, True),
])
def test_step_filter_keeps_feynhiggs_iff_the_tool_exists(mirrors, tool_layout,
                                                         in_env, on_path, kept):
    '''Pins the union semantics itself, not just that the two copies match:
    both-agree-and-both-wrong would otherwise pass the test above.'''
    env_bin = tool_layout(in_env, on_path)
    names = [s["name"] for s in mirrors(env_bin)["_resolve_steps"](STEPS)]
    assert ("feynhiggs" in names) is kept
    # steps with no executable requirement are never touched
    assert names[0] == "prep_input" and names[-1] == "SModelS"


def test_step_filter_agrees_on_this_machine(mirrors, repo):
    '''The real checkout, real PATH: what CI and every developer actually hit.'''
    env_bin = str(repo / ".pixi" / "envs" / "default" / "bin")
    assert mirrors(env_bin)["_resolve_steps"](STEPS) == \
        pipeline.resolve_steps(STEPS, env_bin=env_bin)


# ── batch identity: _batch_seed vs pipeline.batch_seed ─────────────────────

@pytest.mark.parametrize("runid", ["0", "1", "90001", "20260727", "999999999999"])
@pytest.mark.parametrize("base_seed", [0, 123, 2**31 - 1])
@pytest.mark.parametrize("b", [0, 1, 7, 49])
def test_batch_seed_agrees(mirrors, runid, base_seed, b):
    assert mirrors()["_batch_seed"](runid, base_seed, b) == \
        pipeline.batch_seed(runid, base_seed, b)


# ── ntuple naming: _ntuple_basename vs pipeline.ntuple_name ────────────────

@pytest.mark.parametrize("scan_dir", [
    "scan",
    "run/scan_flat",
    "scan.90001.0",
    "batchtest/scan.90001.12",
    "/ptmp/user/runs/scan.20260727.3",
    "scan_with.two.dots.here",          # 4 parts -> falls back to 0.0
    "a.b",                              # 2 parts -> falls back to 0.0
])
def test_ntuple_naming_agrees(mirrors, scan_dir):
    '''The Snakefile derives a basename from the scan NAME, pipeline a full path
    from the scan DIR; composed, they must be the same file.'''
    basename = mirrors()["_ntuple_basename"](os.path.basename(scan_dir))
    assert f"{scan_dir}/{basename}" == pipeline.ntuple_name(scan_dir)
