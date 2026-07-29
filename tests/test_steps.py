'''Tool wrappers: the pMSSM parameter mapping and the workdir isolation fixes.

The external physics binaries are replaced by stubs, so what is exercised here
is exactly our own code: which EXTPAR entries a sampled point lands in, and
whether a tool's fixed-name/side-car output files are kept out of the shared
scan directories (the bug class that made concurrent batches unsafe).
'''

import os

import pyslha
import pytest

from Run3ModelGen import steps


POINT = {
    "M_1": 111.0, "M_2": 222.0, "M_3": 3333.0,
    "AT": -4000.0, "Ab": -444.0, "Atau": -44.0,
    "mu": 555.0, "tanb": 25.0, "mA": 1500.0,
    "meL": 600.0, "mtauL": 2000.0, "meR": 700.0, "mtauR": 2100.0,
    "mqL1": 4000.0, "mqL3": 1200.0, "muR": 4100.0, "mtR": 1300.0,
    "mdR": 4200.0, "mbR": 1400.0,
}


@pytest.fixture
def prepped(tmp_path, raw_slha):
    out = tmp_path / "0.slha"
    assert steps.run_prep_input(point=POINT, rawfilen=raw_slha, outfile=str(out)) is True
    return pyslha.read(str(out), ignorenomass=True).blocks["EXTPAR"]


# ── prep_input: the sampled point -> EXTPAR mapping ────────────────────────

@pytest.mark.parametrize("entry, param", [
    (1, "M_1"), (2, "M_2"), (3, "M_3"),
    (11, "AT"), (12, "Ab"), (13, "Atau"),
    (23, "mu"), (25, "tanb"), (26, "mA"),
    (31, "meL"), (33, "mtauL"), (34, "meR"), (36, "mtauR"),
    (41, "mqL1"), (43, "mqL3"), (44, "muR"), (46, "mtR"),
    (47, "mdR"), (49, "mbR"),
])
def test_prep_input_writes_each_parameter_to_its_extpar_entry(prepped, entry, param):
    assert prepped[entry] == pytest.approx(POINT[param])


@pytest.mark.parametrize("second_gen, first_gen, param", [
    (32, 31, "meL"),    # mmuL := meL
    (35, 34, "meR"),    # mmuR := meR
    (42, 41, "mqL1"),   # mqL2 := mqL1
    (45, 44, "muR"),    # mcR  := muR
    (48, 47, "mdR"),    # msR  := mdR
])
def test_prep_input_ties_second_generation_sfermions_to_the_first(prepped, second_gen, first_gen, param):
    assert prepped[second_gen] == pytest.approx(POINT[param])
    assert prepped[second_gen] == prepped[first_gen]


def test_prep_input_does_not_modify_the_template(raw_slha, tmp_path):
    before = open(raw_slha, "rb").read()
    steps.run_prep_input(point=POINT, rawfilen=raw_slha, outfile=str(tmp_path / "x.slha"))
    assert open(raw_slha, "rb").read() == before


# ── SPheno: immune to deep scan directories, scratch stays private ─────────

SPHENO_OUTPUT = """BLOCK MASS
 25 125.10
 35 1500.0
 36 1500.1
 37 1502.0
 1000006 1200.0
 2000006 1400.0
"""


def test_spheno_is_called_with_short_relative_filenames(tmp_path, stub_bin, raw_slha):
    '''SPheno truncates filename arguments at 120 characters (fixed-length
    Fortran buffers) and then fails with "matrix contains NaN", which is
    indistinguishable from an unphysical point. Running it on short relative
    names inside a private workdir makes scan-directory depth irrelevant.
    '''
    stub_bin("SPheno", 'echo "ARGS:$*" >&2\n'
                       'test ${#1} -le 20 || { echo "path too long for SPheno" >&2; exit 1; }\n'
                       f'cat > "$2" <<EOF\n{SPHENO_OUTPUT}EOF\n')
    # a scan dir far deeper than SPheno's buffer
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    (deep / "SPheno").mkdir(parents=True)
    infile = deep / "in.slha"
    infile.write_text(SPHENO_OUTPUT)
    out = deep / "SPheno" / "0.slha"
    assert len(str(out)) > 120, "the test path must exceed SPheno's buffer to be meaningful"

    ok = steps.run_SPheno(infile=str(infile), outfile=str(out),
                          logfile=str(deep / "0.log"), rawfilen=raw_slha)
    assert ok is True
    assert out.exists()


def test_spheno_scratch_does_not_land_in_the_working_directory(tmp_path, stub_bin, raw_slha, monkeypatch):
    '''SPheno drops files (Messages.out, ...) into its cwd; two concurrent
    models must not share them.'''
    stub_bin("SPheno", 'echo junk > Messages.out\n'
                       f'cat > "$2" <<EOF\n{SPHENO_OUTPUT}EOF\n')
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    infile = tmp_path / "in.slha"
    infile.write_text(SPHENO_OUTPUT)

    steps.run_SPheno(infile=str(infile), outfile=str(tmp_path / "0.slha"),
                     logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert list(cwd.iterdir()) == []


def test_spheno_failure_is_reported_from_the_log(tmp_path, stub_bin, raw_slha):
    stub_bin("SPheno", f'cat > "$2" <<EOF\n{SPHENO_OUTPUT}EOF\n'
                       'echo " The error has occured in the following chain" >&2\n')
    infile = tmp_path / "in.slha"
    infile.write_text(SPHENO_OUTPUT)
    out = tmp_path / "0.slha"
    ok = steps.run_SPheno(infile=str(infile), outfile=str(out),
                          logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert ok is False
    assert out.exists(), ("a failed model's partial output stayed in the scan dir before, "
                          "and the ntuple/tarball contents depend on that")


def test_spheno_reports_failure_when_no_output_appears(tmp_path, stub_bin, raw_slha):
    stub_bin("SPheno", "exit 1\n")
    infile = tmp_path / "in.slha"
    infile.write_text(SPHENO_OUTPUT)
    assert steps.run_SPheno(infile=str(infile), outfile=str(tmp_path / "0.slha"),
                            logfile=str(tmp_path / "0.log"), rawfilen=raw_slha) is False


# ── superiso: fixed-name output must not race in a shared cwd ──────────────

def test_superiso_collects_its_fixed_name_output(tmp_path, stub_bin):
    stub_bin("slha.x", 'echo "BLOCK FOBSERVABLES" > output.flha\necho " 1 0.5" >> output.flha\n')
    infile = tmp_path / "in.slha"
    infile.write_text("BLOCK MASS\n 25 125.0\n")
    out = tmp_path / "superiso" / "0.flha"
    out.parent.mkdir()

    assert steps.run_superiso(infile=str(infile), outfile=str(out)) is True
    assert "FOBSERVABLES" in out.read_text()


def test_superiso_ignores_a_stale_output_flha_in_the_process_cwd(tmp_path, stub_bin, monkeypatch):
    '''slha.x always writes the fixed name output.flha. If it ran in the shared
    process cwd, a file left there by a concurrent model would be harvested as
    THIS model's result — wrong physics, silently. Simulated by a tool that
    produces nothing while a foreign output.flha sits in the cwd.
    '''
    stub_bin("slha.x", "exit 0\n")          # produces no output of its own
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    foreign = cwd / "output.flha"
    foreign.write_text("RESULT OF A DIFFERENT MODEL\n")
    infile = tmp_path / "in.slha"
    infile.write_text("BLOCK MASS\n 25 125.0\n")
    out = tmp_path / "0.flha"

    assert steps.run_superiso(infile=str(infile), outfile=str(out)) is False
    assert not out.exists(), "a foreign output.flha must never become this model's result"
    assert foreign.read_text() == "RESULT OF A DIFFERENT MODEL\n"


def test_superiso_leaves_no_output_flha_in_the_working_directory(tmp_path, stub_bin, monkeypatch):
    '''The tool's scratch must stay in its private workdir.'''
    stub_bin("slha.x", 'echo "x" > output.flha\n')
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    infile = tmp_path / "in.slha"
    infile.write_text("BLOCK MASS\n 25 125.0\n")

    steps.run_superiso(infile=str(infile), outfile=str(tmp_path / "0.flha"))
    assert list(cwd.iterdir()) == []


def test_superiso_reports_failure_when_no_output_is_produced(tmp_path, stub_bin):
    stub_bin("slha.x", "exit 1\n")
    infile = tmp_path / "in.slha"
    infile.write_text("BLOCK MASS\n 25 125.0\n")
    assert steps.run_superiso(infile=str(infile), outfile=str(tmp_path / "0.flha")) is False


def test_superiso_accepts_a_relative_input_path(tmp_path, stub_bin, monkeypatch):
    '''The tool runs in a private cwd, so a relative input must still resolve.'''
    stub_bin("slha.x", 'test -f "$1" || exit 1\necho "ok" > output.flha\n')
    monkeypatch.chdir(tmp_path)
    os.mkdir("SPheno")
    with open("SPheno/0.slha", "w") as f:
        f.write("BLOCK MASS\n 25 125.0\n")
    assert steps.run_superiso(infile="SPheno/0.slha", outfile="0.flha") is True


# ── feynhiggs: side-car output must not land next to the upstream input ────

FH_OUTPUT = """BLOCK MASS
 25 125.10
 35 1500.0
 36 1500.1
 37 1502.0
 1000006 1200.0
 2000006 1400.0
"""


def test_feynhiggs_moves_its_side_car_output_to_the_target(tmp_path, stub_bin, raw_slha):
    stub_bin("FeynHiggs", f'cat > "$1.fh-001" <<EOF\n{FH_OUTPUT}EOF\n')
    spheno = tmp_path / "SPheno"
    spheno.mkdir()
    infile = spheno / "0.slha"
    infile.write_text(FH_OUTPUT)
    out = tmp_path / "feynhiggs" / "0.slha"
    out.parent.mkdir()

    ok = steps.run_feynhiggs(infile=str(infile), outfile=str(out),
                             logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert ok is True
    assert out.exists()


def test_feynhiggs_keeps_the_upstream_spheno_directory_clean(tmp_path, stub_bin, raw_slha):
    '''Pre-fix, FeynHiggs wrote <input>.fh-001 into the SPheno output dir — a
    stray file inside another rule's outputs, left behind even on failure.'''
    stub_bin("FeynHiggs", f'cat > "$1.fh-001" <<EOF\n{FH_OUTPUT}EOF\n')
    spheno = tmp_path / "SPheno"
    spheno.mkdir()
    infile = spheno / "0.slha"
    infile.write_text(FH_OUTPUT)
    out = tmp_path / "feynhiggs" / "0.slha"
    out.parent.mkdir()

    steps.run_feynhiggs(infile=str(infile), outfile=str(out),
                        logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert sorted(p.name for p in spheno.iterdir()) == ["0.slha"]


def test_feynhiggs_failure_leaves_no_side_car_behind(tmp_path, stub_bin, raw_slha):
    stub_bin("FeynHiggs", 'cat > "$1.fh-001" <<EOF\njunk\nEOF\necho "ERROR: bad point" >&2\n')
    spheno = tmp_path / "SPheno"
    spheno.mkdir()
    infile = spheno / "0.slha"
    infile.write_text(FH_OUTPUT)

    ok = steps.run_feynhiggs(infile=str(infile), outfile=str(tmp_path / "0.slha"),
                             logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert ok is False, "an ERROR in the log must fail the step"
    assert sorted(p.name for p in spheno.iterdir()) == ["0.slha"]


def test_feynhiggs_reports_failure_when_the_side_car_is_missing(tmp_path, stub_bin, raw_slha):
    stub_bin("FeynHiggs", "true\n")
    infile = tmp_path / "0.slha"
    infile.write_text(FH_OUTPUT)
    assert steps.run_feynhiggs(infile=str(infile), outfile=str(tmp_path / "out.slha"),
                               logfile=str(tmp_path / "0.log"), rawfilen=raw_slha) is False


# ── micromegas: the csv extraction decides success ────────────────────────

def test_micromegas_failure_when_the_raw_output_has_no_variables(tmp_path, stub_bin):
    stub_bin("main", 'echo "nothing useful here"\n')
    raw = tmp_path / "0.out"
    out = tmp_path / "0.csv"
    ok = steps.run_micromegas(infile=str(tmp_path / "in.slha"),
                              outfile_raw=str(raw), outfile=str(out))
    assert ok is False
    assert not out.exists(), "no csv may be left behind for a failed model"


def test_micromegas_success_writes_the_extracted_csv(tmp_path, stub_bin):
    stub_bin("main", 'echo "  Omega 0.1187"\n')
    raw = tmp_path / "0.out"
    out = tmp_path / "0.csv"
    assert steps.run_micromegas(infile=str(tmp_path / "in.slha"),
                                outfile_raw=str(raw), outfile=str(out)) is True
    assert "Omega,0.1187" in out.read_text()


# ── pathological spectra: NaN masses are a failed model, not a failed run ──

NAN_OUTPUT = """BLOCK MASS
 25 125.10
 35 1500.0
 36 1500.1
 37 1502.0
 1000012                NaN   # MSf(1,1,1)
 1000006                NaN   # MSf(1,3,3)
 2000006                NaN   # MSf(2,3,3)
"""


def test_feynhiggs_nan_masses_fail_only_that_model(tmp_path, stub_bin, raw_slha):
    '''Observed at scale: FeynHiggs writes NaN masses for a few points per 10,000.
    pyslha returns those entries as strings, and multiplying them in
    addinputblocks raised TypeError — which killed the whole batch job. It must
    be an ordinary per-model failure instead.'''
    stub_bin("FeynHiggs", f'cat > "$1.fh-001" <<EOF\n{NAN_OUTPUT}EOF\n')
    infile = tmp_path / "0.slha"
    infile.write_text(FH_OUTPUT)

    ok = steps.run_feynhiggs(infile=str(infile), outfile=str(tmp_path / "out.slha"),
                             logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert ok is False


def test_spheno_nan_masses_fail_only_that_model(tmp_path, stub_bin, raw_slha):
    stub_bin("SPheno", f'cat > "$2" <<EOF\n{NAN_OUTPUT}EOF\n')
    infile = tmp_path / "in.slha"
    infile.write_text(SPHENO_OUTPUT)

    ok = steps.run_SPheno(infile=str(infile), outfile=str(tmp_path / "0.slha"),
                          logfile=str(tmp_path / "0.log"), rawfilen=raw_slha)
    assert ok is False


def test_a_usable_spectrum_still_gets_its_blocks_reinjected(tmp_path, raw_slha):
    '''The guard must not reject healthy spectra.'''
    from Run3ModelGen.addinputblocks import addinputblocks

    good = tmp_path / "good.slha"
    good.write_text(SPHENO_OUTPUT)
    addinputblocks(infile=str(good), blocksfile=raw_slha)
    text = good.read_text()
    for block in ("GM2CALCCONFIG", "SOFTSUSY", "SPHENOINPUT"):
        assert f"BLOCK {block}" in text


# NB: no "#" spellings — SLHA treats it as a comment, so "-1.#IND" parses as -1.
@pytest.mark.parametrize("bad", ["NaN", "*******", "not-a-number", ""])
def test_unusable_stop_masses_are_reported_not_crashed(tmp_path, raw_slha, bad):
    '''Fortran field overflow ("****") and platform NaN spellings must all be
    caught by the same guard rather than raising TypeError deep in numpy.'''
    from Run3ModelGen.addinputblocks import addinputblocks, UnusableSpectrum

    spectrum = tmp_path / "bad.slha"
    spectrum.write_text(NAN_OUTPUT.replace("NaN", bad) if bad else
                        "BLOCK MASS\n 25 125.10\n 35 1500.0\n 36 1500.1\n 37 1502.0\n")
    with pytest.raises(UnusableSpectrum):
        addinputblocks(infile=str(spectrum), blocksfile=raw_slha)
    assert spectrum.read_text().count("BLOCK") >= 1, "the file must be left intact on rejection"


# ── SModelS: the results database is loaded once per process ───────────────

def test_smodels_session_is_cached_per_parameter_file(monkeypatch):
    '''loadDatabase unpickles ~1.4 GB and does not give the memory back, so
    loading it per model grew RSS ~170 MB/model and OOM-killed a 10k batch after
    46 models. The session must be reused; a cache miss would re-import smodels
    and reload, which this test would expose as a raised ImportError.'''
    sentinel = ("parser-sentinel", "database-sentinel")
    monkeypatch.setitem(steps._SMODELS_SESSIONS, "/fake/params.ini", sentinel)

    for _ in range(5):
        assert steps._smodels_session("/fake/params.ini") is sentinel


def test_smodels_sessions_are_keyed_by_parameter_file(monkeypatch):
    '''Two different ini files must not share a database.'''
    monkeypatch.setitem(steps._SMODELS_SESSIONS, "/a.ini", ("pa", "da"))
    monkeypatch.setitem(steps._SMODELS_SESSIONS, "/b.ini", ("pb", "db"))
    assert steps._smodels_session("/a.ini") != steps._smodels_session("/b.ini")
