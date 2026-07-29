'''Tar staging in the batch rule bodies (workflow/scripts/r3mg_batch.py).

The tarball rule stages the scan tree with hardlinks to avoid copying the
data. AFS — the scan_parent on the lxplus/HTCondor backend — refuses hardlinks
between directories even inside one volume, so the link must degrade to a copy
instead of failing the rule. Legacy scans never hit this because they ran on
the worker's local /pool/condor disk.
'''

import errno
import importlib.util
import os

import pytest


def _load_r3mg_batch(repo):
    '''Import the rule-body script by path — it is deliberately not on PATH.'''
    path = repo / "workflow" / "scripts" / "r3mg_batch.py"
    spec = importlib.util.spec_from_file_location("r3mg_batch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def batch(repo):
    return _load_r3mg_batch(repo)


def test_stage_file_links_when_the_filesystem_allows_it(batch, tmp_path):
    src = tmp_path / "src.slha"
    src.write_text("BLOCK MASS\n")
    dst = tmp_path / "dst.slha"

    # tmp_path is normally local disk; skip rather than fail if TMPDIR happens
    # to sit on a filesystem that refuses links (that path is covered below).
    probe = tmp_path / "probe"
    try:
        os.link(str(src), str(probe))
        os.remove(probe)
    except OSError:
        pytest.skip("filesystem refuses hardlinks")

    batch._stage_file(str(src), str(dst))

    assert dst.read_text() == "BLOCK MASS\n"
    assert os.stat(src).st_ino == os.stat(dst).st_ino, "should be a link, not a copy"


@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP])
def test_stage_file_copies_when_links_are_refused(batch, tmp_path, monkeypatch, code):
    '''AFS raises EXDEV here; the tar contents must come out the same anyway.'''
    src = tmp_path / "src.slha"
    src.write_text("XSECTION 1.30E+04\n")
    dst = tmp_path / "dst.slha"

    def _refuse(*args, **kwargs):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(batch.os, "link", _refuse)

    batch._stage_file(str(src), str(dst))

    assert dst.read_text() == "XSECTION 1.30E+04\n"
    assert os.stat(src).st_ino != os.stat(dst).st_ino, "expected a copy"


def test_stage_file_reraises_unrelated_oserrors(batch, tmp_path, monkeypatch):
    '''A full disk must fail the rule, not silently produce a partial tarball.'''
    src = tmp_path / "src.slha"
    src.write_text("x\n")

    def _enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
    monkeypatch.setattr(batch.os, "link", _enospc)

    with pytest.raises(OSError) as e:
        batch._stage_file(str(src), str(tmp_path / "dst.slha"))
    assert e.value.errno == errno.ENOSPC
