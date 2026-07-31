'''Per-step success measured by ntuple content, and the SModelS r-value branches.

Two things a tool's exit status does not tell you, both seen in production:

  * SModelS wrote a result file with no ExptRes (it had been given no cross
    sections) and the step reported "116/116 succeeded" for a scan whose ntuple
    contained zero SModelS branches.
  * SPheno's "charged LSP" SPINFO warning makes pyslha raise, so an otherwise
    complete spectrum yields no SP_* branches at all.

So the reported success rate is now "did this step leave variables the ntuple can
carry", computed with the ntuple's own extraction. Gating is deliberately NOT that
(it stays "the tool ran"), so a stricter statistic cannot shrink the dataset.

The r-value tests pin down the analysis-selection rule, which matters a lot: on real
output, selecting by smallest expected upper limit gave r = 0.004 for a point whose
true maximum r was 2.11 — i.e. excluded, reported three orders of magnitude below the
exclusion threshold.
'''
import textwrap

import pytest

from Run3ModelGen import ntupling


SMODELS_STEP = {"name": "SModelS", "output_dir": "SModelS", "prefix": "SModelS",
                "input_dir": "SPheno"}
SPHENO_STEP = {"name": "SPheno", "output_dir": "SPheno", "prefix": "SP",
               "input_dir": "input"}


def _write_smodels(tmp_path, body, modelnum=0):
    d = tmp_path / "SModelS"
    d.mkdir(exist_ok=True)
    (d / f"{modelnum}.slha.py").write_text(textwrap.dedent(body))
    return str(tmp_path)


def _read(scan_dir, step=SMODELS_STEP, modelnum=0):
    inputdef, _ = ntupling.step_ntuple_spec(step)
    data = ntupling.readModel([inputdef], modelnum, scan_dir, "x.0.0.root", False)
    data.pop("model", None)
    return data


# ── step_ntuple_spec: one source of truth for reader + success test ──────────────

@pytest.mark.parametrize("name,filetype", [
    ("prep_input", ".slha"), ("SPheno", ".slha"), ("softsusy", ".slha"),
    ("micromegas", ".csv"), ("superiso", ".flha"), ("gm2calc", ".slha"),
    ("feynhiggs", ".slha"), ("SModelS", ".slha.py"),
])
def test_spec_filetype_per_step(name, filetype):
    step = {"name": name, "output_dir": name, "prefix": "X"}
    (directory, ft, prefix, _blocks, _decays, _needs), _required = \
        ntupling.step_ntuple_spec(step)
    assert ft == filetype and directory == f"{name}/" and prefix == "X"


def test_spec_rejects_unknown_step():
    with pytest.raises(KeyError):
        ntupling.step_ntuple_spec({"name": "nosuchtool", "output_dir": "x", "prefix": "X"})


# ── success = produced ntuple variables ─────────────────────────────────────────

def test_missing_output_is_not_a_success(tmp_path):
    (tmp_path / "SModelS").mkdir()
    assert ntupling.step_produced_ntuple_vars(SMODELS_STEP, 0, str(tmp_path)) is False


def test_smodels_without_exptres_is_not_a_success(tmp_path):
    '''The exact production case: no cross sections -> file status -4, no ExptRes.'''
    scan = _write_smodels(tmp_path, """
        smodelsOutput = {
        'OutputStatus' : {
            'file status' : -4,
            'warnings' : 'Error checking SLHA file: No cross-sections found',
        }
        }
        """)
    assert ntupling.step_produced_ntuple_vars(SMODELS_STEP, 0, scan) is False


def test_smodels_with_results_is_a_success(tmp_path):
    scan = _write_smodels(tmp_path, """
        smodelsOutput = {
        'OutputStatus' : {'file status' : 1},
        'ExptRes' : [
            {'AnalysisID': 'CMS-SUS-19-010', 'theory prediction (fb)': 10.3,
             'upper limit (fb)': 738.7, 'expected upper limit (fb)': 595.3,
             'r': 0.0139, 'r_expected': 0.0173},
        ]}
        """)
    assert ntupling.step_produced_ntuple_vars(SMODELS_STEP, 0, scan) is True


def test_unparseable_spectrum_is_not_a_success(tmp_path):
    '''An SPINFO warning entry makes pyslha raise, so no SP_* branches exist. The
    spectrum may be physically fine, but it contributes nothing to the ntuple.'''
    d = tmp_path / "SPheno"
    d.mkdir()
    (d / "0.slha").write_text(textwrap.dedent("""
        Block SPINFO         # Program information
             1   SPheno      # spectrum calculator
             2   v4.0.7      # version number
             3               # charged LSP, please check block MASS for its identity
        Block MASS
             25     1.23236121E+02  # h0
        """))
    assert ntupling.step_produced_ntuple_vars(SPHENO_STEP, 0, str(tmp_path)) is False


def test_unknown_step_is_not_reported_as_failure(tmp_path):
    step = {"name": "nosuchtool", "output_dir": "x", "prefix": "X"}
    assert ntupling.step_produced_ntuple_vars(step, 0, str(tmp_path)) is True


# ── r-value branches ────────────────────────────────────────────────────────────

# Three analyses. The most sensitive by SMALLEST expected UL is 'tiny' (its limit is
# strongest) but almost no signal is predicted there; the exclusion actually comes
# from 'excluder'. This is the real shape of the production case.
_THREE = """
    smodelsOutput = {
    'OutputStatus' : {'file status' : 1},
    'ExptRes' : [
        {'AnalysisID': 'A-tiny',     'theory prediction (fb)': 0.01,
         'upper limit (fb)': 300.0,  'expected upper limit (fb)': 1.0,
         'r': 0.00003, 'r_expected': 0.01},
        {'AnalysisID': 'A-excluder', 'theory prediction (fb)': 200.0,
         'upper limit (fb)': 95.0,   'expected upper limit (fb)': 90.0,
         'r': 2.10933, 'r_expected': 2.2},
        {'AnalysisID': 'A-middle',   'theory prediction (fb)': 5.0,
         'upper limit (fb)': 50.0,   'expected upper limit (fb)': 40.0,
         'r': 0.1, 'r_expected': 0.125},
    ]}
    """


def test_r_max_is_the_exclusion_verdict(tmp_path):
    d = _read(_write_smodels(tmp_path, _THREE))
    assert d["SModelS_r_max"] == pytest.approx(2.10933)
    assert d["SModelS_r_max"] > 1, "this point is excluded and must be visible as such"


def test_best_expected_r_selection_finds_the_excluder(tmp_path):
    d = _read(_write_smodels(tmp_path, _THREE))
    assert d["SModelS_bestExpR_r"] == pytest.approx(2.10933)
    assert d["SModelS_bestExpR_r_expected"] == pytest.approx(2.2)


def test_smallest_expected_ul_selection_misses_it(tmp_path):
    '''Regression guard on the distinction, not an endorsement: bestExpUL selects
    A-tiny, whose r is 5 orders of magnitude below the exclusion.'''
    d = _read(_write_smodels(tmp_path, _THREE))
    assert d["SModelS_bestExpUL_r"] == pytest.approx(0.00003)
    assert d["SModelS_bestExpUL_ExpUpperLimit"] == pytest.approx(1.0)
    assert d["SModelS_bestExpUL_r"] < 1 < d["SModelS_r_max"]


def test_r_falls_back_to_the_ratio_when_absent(tmp_path):
    '''Older SModelS output may not carry 'r'; derive it rather than drop the branch.'''
    scan = _write_smodels(tmp_path, """
        smodelsOutput = {
        'OutputStatus' : {'file status' : 1},
        'ExptRes' : [
            {'AnalysisID': 'A', 'theory prediction (fb)': 50.0,
             'upper limit (fb)': 100.0, 'expected upper limit (fb)': 200.0},
        ]}
        """)
    d = _read(scan)
    assert d["SModelS_bestExpUL_r"] == pytest.approx(0.5)
    assert d["SModelS_bestExpUL_r_expected"] == pytest.approx(0.25)


def test_no_exptres_yields_no_r_branches(tmp_path):
    scan = _write_smodels(tmp_path, """
        smodelsOutput = {'OutputStatus' : {'file status' : -4}}
        """)
    d = _read(scan)
    assert not [k for k in d if "_r" in k]
