'''The `hard_window` constraint type and micromegas' LSP-identity flag.

Both exist to express requirements the numeric chi2 constraints could not: "this
observable must lie in this range, full stop" and "the LSP must be the lightest
neutralino". Before them, a relic-density scan whose only constraint was a Gaussian on
Omega accepted ~51% sneutrino/slepton-LSP spectra, because micrOMEGAs computes the
abundance of whatever the lightest odd particle happens to be and nothing in the
likelihood said it had to be a neutralino.
'''
import math

import pytest

from Run3ModelGen.microextract import microextract
from Run3ModelGen.modelgen import ModelGenerator


# ── hard_window arithmetic ──────────────────────────────────────────────────────

_term = ModelGenerator._compute_chi2_term


def test_hard_window_inside_costs_nothing():
    c = {"type": "hard_window", "lo": 0, "hi": 1.0}
    assert _term(0.12, c) == 0.0


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0, 5.0])
def test_hard_window_outside_is_infinite(value):
    # bounds are strict, so the endpoints themselves are vetoed
    c = {"type": "hard_window", "lo": 0, "hi": 1.0}
    assert math.isinf(_term(value, c))


def test_hard_window_floor_only():
    c = {"type": "hard_window", "lo": 0}          # e.g. m_h > 0 (reject the -1 sentinel)
    assert _term(123.0, c) == 0.0
    assert math.isinf(_term(-1.0, c))


def test_hard_window_as_boolean_flag():
    c = {"type": "hard_window", "lo": 0.5}        # cdm_is_neutralino == 1
    assert _term(1.0, c) == 0.0
    assert math.isinf(_term(0.0, c))


def test_hard_window_needs_no_central_or_uncertainty():
    # Regression: the old implementation read constraint['central'] unconditionally.
    _term(0.5, {"type": "hard_window", "lo": 0, "hi": 1})


def test_other_types_still_work():
    g = {"type": "gaussian", "central": 0.12, "uncertainty": 0.012}
    assert _term(0.12, g) == 0.0
    assert _term(0.132, g) == pytest.approx(1.0)
    ub = {"type": "upper_bound", "central": 0.12, "uncertainty": 0.012}
    assert _term(0.10, ub) == 0.0 and _term(0.132, ub) == pytest.approx(1.0)


# ── config validation ───────────────────────────────────────────────────────────

def _mg_with_constraints(constraints):
    mg = ModelGenerator.__new__(ModelGenerator)      # no config/pipeline needed
    mg.prior = 'mcmc'
    mg.steps = [{"name": "micromegas"}]
    mg.constraints = constraints
    mg.mcmc = {}
    return mg


def test_hard_window_without_bounds_is_rejected():
    mg = _mg_with_constraints([{"name": "x", "type": "hard_window",
                                "source": {"step": "micromegas"}}])
    with pytest.raises(ValueError, match="neither 'lo' nor 'hi'"):
        mg._validate_observable_sources()


def test_empty_hard_window_is_rejected():
    mg = _mg_with_constraints([{"name": "x", "type": "hard_window", "lo": 1, "hi": 1,
                                "source": {"step": "micromegas"}}])
    with pytest.raises(ValueError, match="empty hard_window"):
        mg._validate_observable_sources()


def test_hard_window_does_not_need_uncertainty():
    mg = _mg_with_constraints([{"name": "x", "type": "hard_window", "lo": 0, "hi": 1,
                                "source": {"step": "micromegas"}}])
    mg._validate_observable_sources()      # must not raise about uncertainty


def test_gaussian_still_requires_positive_uncertainty():
    mg = _mg_with_constraints([{"name": "x", "type": "gaussian", "central": 1,
                                "uncertainty": 0, "source": {"step": "micromegas"}}])
    with pytest.raises(ValueError, match="non-positive uncertainty"):
        mg._validate_observable_sources()


# ── micromegas LSP-identity extraction ──────────────────────────────────────────

_RAW_HEAD = """
========= SLHA file input =========
Warnings from spectrum calculator:
"""
# tab-separated name/value, exactly as micrOMEGAs writes it (microextract tokenises
# on whitespace and reads the token after the name)
_TAIL = "Xf\t2.67e+01 Omega\t1.20e-01\n"


def _extract(tmp_path, candidate_line):
    raw = tmp_path / "raw.txt"
    raw.write_text(_RAW_HEAD + candidate_line + _TAIL)
    out = tmp_path / "out.csv"
    ok = microextract(infilen=str(raw), outfilen=str(out))
    assert ok
    return dict(l.split(",", 1) for l in out.read_text().splitlines()[1:])


def test_neutralino_lsp_flagged_one(tmp_path):
    d = _extract(tmp_path, "Dark matter candidate is '~o1' with spin=1/2  mass=1.18E+03\n")
    assert d["cdm_is_neutralino"] == "1"
    assert d["cdm_mass"] == "1.18E+03"


def test_sneutrino_lsp_flagged_zero(tmp_path):
    # the real line from the run that motivated this: a 644 GeV mu-sneutrino LSP
    d = _extract(tmp_path, "Dark matter candidate is '~nm' with spin=0/2  mass=6.44E+02\n")
    assert d["cdm_is_neutralino"] == "0"


def test_charged_slepton_lsp_flagged_zero(tmp_path):
    d = _extract(tmp_path, "Dark matter candidate is '~eR' with spin=0/2  mass=5.00E+02\n")
    assert d["cdm_is_neutralino"] == "0"


def test_missing_candidate_line_omits_the_key(tmp_path):
    '''Fail closed: with no identity line the key is absent, so a hard_window on it
    makes the observable unevaluable and the point is dropped rather than accepted.'''
    d = _extract(tmp_path, "")
    assert "cdm_is_neutralino" not in d
    assert "Omega" in d, "the rest of the extraction must be unaffected"


# ── the plot target must not be a veto ──────────────────────────────────────────

def test_target_skips_hard_window_vetoes(tmp_path, repo):
    '''Regression: v4 listed `higgs_computed` (m_h > 0, a hard_window) first, so every
    input-vs-target diagnostic plot came out against m_h instead of the relic density.
    A veto carves out the support; it is not what the posterior is fitted to.'''
    import importlib.util
    import yaml

    path = repo / "source" / "Run3ModelGen" / "scripts" / "emcee_diagnostics.py"
    spec = importlib.util.spec_from_file_location("emcee_diagnostics_t", path)
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)

    om = {"step": "micromegas", "file_type": "csv", "variable": "Omega"}
    mh = {"step": "SPheno", "file_type": "slha", "block": "MASS", "entry": 25}
    cfg = {
        "constraints": [
            {"name": "higgs_computed", "type": "hard_window", "lo": 0, "source": mh},
            {"name": "relic_density", "type": "gaussian", "central": 0.120,
             "uncertainty": 0.012, "source": om},
            {"name": "relic_subcritical", "type": "hard_window", "lo": 0, "hi": 1.0,
             "source": om},
        ],
        "mcmc": {"record_observables": [{"name": "m_h", "branch": "SP_m_h", "source": mh},
                                        {"name": "Omega", "branch": "MO_Omega", "source": om}]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    meta = {"config_file": str(cfg_path), "record_names": ["m_h", "Omega"]}

    assert diag.infer_target_obs(meta) == "Omega", "a hard_window veto must not win"
    assert diag.infer_target_obs(meta, override="m_h") == "m_h", "--target still wins"

    # All-veto config: fall back rather than return nothing.
    cfg["constraints"] = [c for c in cfg["constraints"] if c["type"] == "hard_window"]
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert diag.infer_target_obs(meta) in ("m_h", "Omega")
