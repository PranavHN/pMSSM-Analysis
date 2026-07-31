'''Cross-section computation for the SModelS step.

Regression tests for a silent, total failure of the SModelS step. Two bugs on the
same few lines:

1. Cross sections were computed at ``maxOrder=NLL`` only. NLL k-factors come from
   NLLfast, which covers ONLY gluino and squark production, so a spectrum whose light
   states are electroweakinos and sleptons got NO cross sections at all. Measured on a
   real pMSSM point (gluino and all squarks ~4 TeV, charginos/sleptons ~190 GeV):
   NLL returned 0 cross sections, LO returned 12 — chi_2^0 chi_1^+ at 0.96 pb among
   them. SModelS then reported ``file status: -4`` / "No cross-sections found",
   produced no ExptRes, and the ntuple got 0 SModelS branches.

2. The guard meant to catch that was ``if xsecsNLL == []``. ``compute()`` returns an
   ``XSectionList``; an EMPTY one has ``len() == 0`` and ``bool() is False`` but
   compares ``!= []``, so the guard never fired and the step returned True. The
   manifest recorded "SModelS 116/116 succeeded" for a scan in which every single
   output was unusable.

These tests fake the computer, so they need no results database and no Pythia.
'''
import pytest

from Run3ModelGen import steps


@pytest.fixture
def fake_xseccomputer(monkeypatch):
    '''Replace smodels' XSecComputer with a recording stub.'''
    from smodels.tools import xsecComputer
    from smodels.base.crossSection import XSectionList

    class Fake:
        built = []          # (maxOrder, nevents) per construction
        written = []        # (order, n) per addXSecToFile
        result = None       # what compute() returns; set per test
        raise_for = ()      # maxOrders whose compute() should raise

        def __init__(self, maxOrder, nevents, pythiaVersion):
            self.maxOrder = maxOrder
            type(self).built.append((maxOrder, nevents))

        def compute(self, sqrts, slhafile):
            if self.maxOrder in type(self).raise_for:
                raise RuntimeError(f"boom at order {self.maxOrder}")
            res = type(self).result
            return res(self.maxOrder) if callable(res) else XSectionList()

        def addXSecToFile(self, xsecs, path):
            type(self).written.append((self.maxOrder, len(xsecs)))

    Fake.built, Fake.written, Fake.raise_for, Fake.result = [], [], (), None
    monkeypatch.setattr(xsecComputer, "XSecComputer", Fake)

    # Past the xsec stage run_SModelS loads the results database and runs the real
    # matcher; stub both so these stay unit tests (no 1.4 GB database, no Pythia).
    from smodels.matching import modelTester

    class _Sink:
        def flush(self):
            pass

    monkeypatch.setattr(steps, "_smodels_session", lambda ini: (object(), object()))
    monkeypatch.setattr(modelTester, "testPoint",
                        lambda f, out, parser, db: {"py": _Sink()})
    return Fake


def _nonempty(_order):
    '''An XSectionList that is not empty (one entry is enough for the guard).'''
    from smodels.base.crossSection import XSectionList, XSection
    xs = XSectionList()
    xs.xSections.append(XSection())     # note: lowercase x — `XSections` does not exist
    return xs


def _args(tmp_path):
    src = tmp_path / "in.slha"
    src.write_text("BLOCK MASS\n 1000022 1.0E+02 # ~chi_10\n")
    return dict(infile=str(src), output_dir=str(tmp_path),
                smodels_ini=str(tmp_path / "params.ini"),
                xsec_dest=str(tmp_path / "xsec.slha"))


# ── the guard that never fired ──────────────────────────────────────────────────

def test_empty_xsectionlist_does_not_equal_empty_list():
    '''Documents the exact reason the original `== []` guard was dead code.'''
    from smodels.base.crossSection import XSectionList
    xs = XSectionList()
    assert len(xs) == 0 and not xs
    assert not (xs == []), "if this ever becomes True, the old guard would have worked"


def test_returns_false_when_no_xsecs(tmp_path, fake_xseccomputer):
    fake_xseccomputer.result = None          # empty XSectionList at every order
    assert steps.run_SModelS(**_args(tmp_path)) is False


def test_writes_nothing_when_no_xsecs(tmp_path, fake_xseccomputer):
    fake_xseccomputer.result = None
    steps.run_SModelS(**_args(tmp_path))
    assert fake_xseccomputer.written == [], "must not write an xsec-less file"
    assert not (tmp_path / "xsec.slha").exists(), "must not even create the copy"


# ── LO must always be computed ──────────────────────────────────────────────────

def test_lo_is_always_computed(tmp_path, fake_xseccomputer):
    from smodels.tools.xsecComputer import LO
    fake_xseccomputer.result = _nonempty
    steps.run_SModelS(**_args(tmp_path))
    orders = [o for o, _ in fake_xseccomputer.built]
    assert LO in orders, "LO covers electroweakino/slepton production; it is not optional"
    assert orders[0] == LO, "LO first: its failure is the fatal one"


def test_nll_added_by_default_and_suppressible(tmp_path, fake_xseccomputer):
    from smodels.tools.xsecComputer import LO, NLL
    fake_xseccomputer.result = _nonempty
    steps.run_SModelS(**_args(tmp_path))
    assert [o for o, _ in fake_xseccomputer.built] == [LO, NLL]

    fake_xseccomputer.built = []
    steps.run_SModelS(**_args(tmp_path), include_nll=False)
    assert [o for o, _ in fake_xseccomputer.built] == [LO]


def test_nevents_is_forwarded(tmp_path, fake_xseccomputer):
    fake_xseccomputer.result = _nonempty
    steps.run_SModelS(**_args(tmp_path), nevents=1234)
    assert {n for _, n in fake_xseccomputer.built} == {1234}


# ── failure handling per order ──────────────────────────────────────────────────

def test_lo_failure_is_fatal(tmp_path, fake_xseccomputer):
    from smodels.tools.xsecComputer import LO
    fake_xseccomputer.result = _nonempty
    fake_xseccomputer.raise_for = (LO,)
    assert steps.run_SModelS(**_args(tmp_path)) is False


def test_nll_failure_degrades_to_lo(tmp_path, fake_xseccomputer):
    '''An NLLfast hiccup must not throw away perfectly good LO cross sections.'''
    from smodels.tools.xsecComputer import LO, NLL
    fake_xseccomputer.result = _nonempty
    fake_xseccomputer.raise_for = (NLL,)
    steps.run_SModelS(**_args(tmp_path))          # reaches SModelS, so it got past the guard
    assert [o for o, _ in fake_xseccomputer.written] == [LO]


# ── the throughput setting that motivated turning SR combination off ────────────

def test_combine_srs_is_off_in_the_shipped_ini(repo):
    '''combineSRs dominates per-point runtime and none of it reaches the ntuple.'''
    ini = (repo / "source" / "Run3ModelGen" / "data" / "SModelS_parameters.ini").read_text()
    line = next(l for l in ini.splitlines() if l.strip().startswith("combineSRs"))
    assert line.split(";")[0].split("=")[1].strip() == "False"
