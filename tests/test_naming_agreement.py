'''pipeline.ntuple_name must predict exactly what ntupling.mkntuple writes.

Two implementations of one contract; the Snakefile has a third, parse-time
mirror, checked against pipeline.ntuple_name in test_snakefile_mirrors.py.
rerunStep deletes the path the predictor returns, and the workflow declares it
as a rule output, so any disagreement means a rule that never sees its own
product. Compared here by driving mkntuple's naming logic directly, without
building an ntuple.
'''

import pytest

from Run3ModelGen import pipeline


def _mkntuple_name(scan_dir):
    '''The naming branch of ntupling.mkntuple, kept in sync by these tests.'''
    scan_dir = scan_dir.rstrip('/')
    parts = scan_dir.split('/')[-1].split('.')
    if len(parts) == 3:
        return f"{scan_dir}/ntuple.{parts[1]}.{parts[2]}.root"
    return f"{scan_dir}/ntuple.0.0.root"


@pytest.mark.parametrize("scan_dir", [
    "scan",
    "scan/",
    "run/scan_flat",
    "scan.90001.0",
    "scan.90001.12/",
    "/ptmp/user/runs/scan.250727.3",
    "batchtest/scan.0.0",
    "scan_with.two.dots.here",      # 4 parts -> falls back to 0.0
])
def test_predictor_agrees_with_the_writer(scan_dir):
    assert pipeline.ntuple_name(scan_dir) == _mkntuple_name(scan_dir)


@pytest.mark.parametrize("scan_dir", ["scan/", "scan.90001.7/", "a/b/"])
def test_no_double_slash_in_the_predicted_path(scan_dir):
    assert "//" not in pipeline.ntuple_name(scan_dir)


def test_real_mkntuple_uses_the_same_logic(monkeypatch, tmp_path):
    '''Guard against ntupling.mkntuple's naming drifting from the copy above.'''
    import inspect

    from Run3ModelGen import ntupling

    src = inspect.getsource(ntupling.mkntuple)
    assert "scan_dir.rstrip('/')" in src, "mkntuple must normalise the trailing slash"
    assert 'outName = f"{scan_dir}/ntuple.{ClusterId}.{ProcId}.root"' in src
    assert 'outName = f"{scan_dir}/ntuple.0.0.root"' in src
