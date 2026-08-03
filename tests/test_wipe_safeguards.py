'''Safeguards around the fresh-scan wipe.

Regression tests for a real incident: a *dumped* `<scan_dir>/scan_config.yaml` was
reused as an input `--config_file`. That dump records `scan_dir` and `seed` as
provenance, the ModelGenerator applied every config key as an attribute AFTER the
caller's arguments, so the recorded scan_dir silently won — and the fresh-scan wipe
(`os.system(f"rm -r {dir} &>/dev/null")`, which discarded both the path and any
error) then deleted a completed emcee production scan: chain.h5, its ntuple and its
diagnostics.

Three independent guards, one test group each:
  1. config keys never override the caller's scan_dir/seed;
  2. the wipe refuses to destroy sampler state (chain.h5) that would cost days of
     compute to rebuild (it is reproducible from its seed, just not cheaply);
  3. the wipe is visible in the log and does not use a silent shell rm.
'''
import os

import pytest
import yaml

from Run3ModelGen import pipeline
from Run3ModelGen.modelgen import ModelGenerator


# ── 1. caller-owned scan_dir / seed ─────────────────────────────────────────────

def _dumped_config(tmp_path, scan_dir, seed):
    '''A minimal stand-in for a _dump_scan_config() output: a real config plus the
    provenance keys the dump adds.'''
    cfg = {
        "prior": "flat", "num_models": 2, "isGMSB": False,
        "parameters": {"tanb": [1, 60], "M_1": [-2000, 2000], "M_2": [-2000, 2000],
                       "M_3": [4000, 4000], "AT": [-8000, 8000], "Ab": [-2000, 2000],
                       "Atau": [-2000, 2000], "mu": [-2000, 2000], "mA": [2000, 2000],
                       "meL": [0, 2000], "meR": [0, 2000], "mqL1": [4000, 4000],
                       "mqL3": [4000, 4000], "mtR": [4000, 4000], "mtauL": [2000, 2000],
                       "mtauR": [2000, 2000], "muR": [4000, 4000], "mbR": [4000, 4000],
                       "mdR": [4000, 4000]},
        "steps": [{"name": "prep_input", "output_dir": "input", "prefix": "IN"}],
        # the provenance keys that caused the incident
        "scan_dir": str(scan_dir), "seed": seed,
    }
    path = tmp_path / "dumped_scan_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_config_scan_dir_does_not_override_caller(tmp_path):
    # The config names a scan dir the caller did NOT ask for; the caller must win.
    poisoned = tmp_path / "production_do_not_touch"
    cfg = _dumped_config(tmp_path, poisoned, seed=999)
    mg = ModelGenerator(config_file=cfg, scan_dir=str(tmp_path / "mine"), seed=7, quiet=True)
    assert mg.scan_dir == str(tmp_path / "mine")
    assert str(poisoned) not in str(mg.scan_dir)


def test_config_seed_does_not_override_caller(tmp_path):
    cfg = _dumped_config(tmp_path, tmp_path / "elsewhere", seed=999)
    mg = ModelGenerator(config_file=cfg, scan_dir=str(tmp_path / "mine"), seed=7, quiet=True)
    assert mg.seed == 7


def test_ordinary_config_without_provenance_keys_still_works(tmp_path):
    # The keys only appear in dumps, so a hand-written config must be unaffected.
    cfg = yaml.safe_load(open(_dumped_config(tmp_path, tmp_path / "x", seed=1)))
    del cfg["scan_dir"], cfg["seed"]
    path = tmp_path / "handwritten.yaml"
    path.write_text(yaml.safe_dump(cfg))
    mg = ModelGenerator(config_file=str(path), scan_dir=str(tmp_path / "mine"), seed=7,
                        quiet=True)
    assert mg.scan_dir == str(tmp_path / "mine") and mg.seed == 7


# ── 2. the wipe refuses to destroy sampler state ────────────────────────────────

def test_wipe_refuses_when_chain_h5_present(tmp_path):
    d = tmp_path / "scan"
    d.mkdir()
    (d / "chain.h5").write_bytes(b"not really hdf5, but it is the file that matters")
    (d / "ntuple.0.0.root").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="refusing to wipe"):
        pipeline.wipe_scan_dir(str(d))
    assert (d / "chain.h5").exists(), "guard raised but the chain was deleted anyway"


def test_wipe_allows_ordinary_scan_dir(tmp_path):
    # No chain.h5 -> flat/grid/fixed fresh-scan semantics are unchanged.
    d = tmp_path / "scan"
    (d / "input").mkdir(parents=True)
    (d / "ntuple.0.0.root").write_bytes(b"x")
    pipeline.wipe_scan_dir(str(d))
    assert not d.exists()


def test_wipe_force_flag_overrides(tmp_path):
    d = tmp_path / "scan"
    d.mkdir()
    (d / "chain.h5").write_bytes(b"x")
    pipeline.wipe_scan_dir(str(d), force=True)
    assert not d.exists()


def test_wipe_env_var_overrides(tmp_path, monkeypatch):
    d = tmp_path / "scan"
    d.mkdir()
    (d / "chain.h5").write_bytes(b"x")
    monkeypatch.setenv(pipeline.FORCE_WIPE_ENV, "1")
    pipeline.wipe_scan_dir(str(d))
    assert not d.exists()


def test_wipe_missing_dir_is_a_noop(tmp_path):
    pipeline.wipe_scan_dir(str(tmp_path / "never_existed"))   # must not raise


# ── 3. the wipe is visible ──────────────────────────────────────────────────────

def test_wipe_logs_the_path(tmp_path, caplog):
    d = tmp_path / "scan"
    (d / "input").mkdir(parents=True)
    with caplog.at_level("INFO"):
        pipeline.wipe_scan_dir(str(d))
    assert str(d) in caplog.text, "the deleted path must appear in the log"


def test_no_silent_shell_rm_remains(repo):
    '''The old form discarded the path and any error, so a wipe of the wrong
    directory left nothing in the log to notice.'''
    for rel in ("source/Run3ModelGen/python/modelgen.py",
                "workflow/scripts/r3mg_batch.py"):
        txt = (repo / rel).read_text()
        assert "rm -r " not in txt, f"{rel} still shells out to rm -r for the wipe"
        assert "&>/dev/null" not in txt, f"{rel} still silences the wipe"
