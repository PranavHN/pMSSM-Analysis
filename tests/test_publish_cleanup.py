'''Publishing is a MOVE: the local scan dir goes away, but only if EOS has it.

`--publish` (config keep_local=0) exists so a production run costs nothing
outside $EOSPATH — 300 batches would otherwise leave ~340 GB of scan dirs on
the shared filesystem. The delete is therefore guarded: the rule body re-checks
every published file for presence AND byte size, and keeps the local copies if
anything is off. These tests pin the guard, because the failure mode it
prevents is the irreversible one.
'''

import importlib.util
import os

import pytest
from click.testing import CliRunner


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


@pytest.fixture
def pair(tmp_path):
    '''A (src, dst) copy whose destination the test can then damage.'''
    src = tmp_path / "scan.1.0.tar.gz"
    src.write_bytes(b"payload" * 100)
    dst = tmp_path / "eos" / "scan.1.0.tar.gz"
    dst.parent.mkdir()
    dst.write_bytes(src.read_bytes())
    return str(src), str(dst)


# ── the verification predicate ─────────────────────────────────────────────

def test_a_faithful_copy_verifies(batch, pair):
    assert batch._unverified([pair]) == []


def test_a_missing_destination_does_not_verify(batch, pair):
    os.remove(pair[1])
    assert len(batch._unverified([pair])) == 1


def test_a_truncated_destination_does_not_verify(batch, pair):
    '''A short write (quota, interrupted transfer) must not read as success.'''
    with open(pair[1], "wb") as f:
        f.write(b"payload")
    problems = batch._unverified([pair])
    assert len(problems) == 1 and "expected" in problems[0]


def test_a_destination_directory_does_not_verify(batch, pair):
    '''os.path.getsize would happily size a directory — isfile must gate it.'''
    os.remove(pair[1])
    os.mkdir(pair[1])
    assert len(batch._unverified([pair])) == 1


def test_an_unreadable_remote_size_does_not_verify(batch, pair, monkeypatch):
    '''Fail CLOSED: xrdfs erroring out must never be read as "copy is fine".'''
    monkeypatch.setattr(batch, "_remote_size", lambda url: None)
    remote = (pair[0], "root://eosuser.cern.ch//eos/user/x/scan.1.0.tar.gz")
    assert len(batch._unverified([remote])) == 1


def test_every_bad_destination_is_reported(batch, tmp_path, pair):
    '''The message must name all of them, not just the first.'''
    other_src = tmp_path / "ntuple.1.0.root"
    other_src.write_bytes(b"root")
    os.remove(pair[1])
    problems = batch._unverified([pair, (str(other_src), str(tmp_path / "eos" / "gone.root"))])
    assert len(problems) == 2


# ── _remote_size parsing ───────────────────────────────────────────────────

def test_remote_size_parses_xrdfs_stat(batch, monkeypatch):
    class _Res:
        returncode = 0
        stdout = "Path:   /eos/user/x/f.root\nId:     0\nSize:   4096\nMTime:  now\n"
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: _Res())
    assert batch._remote_size("root://eosuser.cern.ch//eos/user/x/f.root") == 4096


def test_remote_size_is_none_when_xrdfs_fails(batch, monkeypatch):
    class _Res:
        returncode = 1
        stdout = ""
    monkeypatch.setattr(batch.subprocess, "run", lambda *a, **k: _Res())
    assert batch._remote_size("root://eosuser.cern.ch//eos/user/x/f.root") is None


def test_remote_size_is_none_for_a_malformed_url(batch):
    '''No host/path split -> no stat to make; must not crash or guess.'''
    assert batch._remote_size("root://nohostpath") is None


# ── the publish command itself ─────────────────────────────────────────────

def _make_batch_dir(tmp_path, runid="1", procid="0"):
    '''A minimal published-shaped batch: scan dir + ntuple + config + tarball.'''
    scan = tmp_path / f"scan.{runid}.{procid}"
    (scan / "SPheno").mkdir(parents=True)
    (scan / "SPheno" / "0.slha").write_text("BLOCK MASS\n")
    (scan / f"ntuple.{runid}.{procid}.root").write_bytes(b"rootfile")
    (scan / "scan_config.yaml").write_text("prior: flat\n")
    tar = tmp_path / f"scan.{runid}.{procid}.tar.gz"
    tar.write_bytes(b"tarball")
    return scan, tar


def _run_publish(batch, scan, tar, eospath, runid="1", procid="0", cleanup=True):
    args = ["--scan_dir", str(scan), "--tarball", str(tar),
            "--runid", runid, "--procid", procid]
    if cleanup:
        args.append("--cleanup")
    return CliRunner().invoke(batch.cli, ["publish"] + args,
                              env={"EOSPATH": str(eospath)})


def test_publish_with_cleanup_leaves_only_the_eos_copies(batch, tmp_path, monkeypatch):
    scan, tar = _make_batch_dir(tmp_path)
    eos = tmp_path / "eos"
    monkeypatch.setenv("EOSPATH", str(eos))

    res = _run_publish(batch, scan, tar, eos)

    assert res.exit_code == 0, res.output
    assert not scan.exists(), "the scan dir must be gone"
    assert not tar.exists(), "the tarball must be gone"
    assert (eos / "1" / "0" / "scan.1.0.tar.gz").is_file()
    assert (eos / "1" / "0" / "ntuple.1.0.root").is_file()
    assert (eos / "1" / "scan_config.yaml").is_file()


def test_publish_without_cleanup_keeps_both(batch, tmp_path, monkeypatch):
    scan, tar = _make_batch_dir(tmp_path)
    eos = tmp_path / "eos"
    monkeypatch.setenv("EOSPATH", str(eos))

    res = _run_publish(batch, scan, tar, eos, cleanup=False)

    assert res.exit_code == 0, res.output
    assert scan.is_dir() and tar.is_file()
    assert (eos / "1" / "0" / "ntuple.1.0.root").is_file()


def test_publish_refuses_to_delete_when_a_copy_is_damaged(batch, tmp_path, monkeypatch):
    '''The whole point of the guard: a bad copy must cost a failed rule, never
    the only surviving version of the data.'''
    scan, tar = _make_batch_dir(tmp_path)
    eos = tmp_path / "eos"
    monkeypatch.setenv("EOSPATH", str(eos))

    real_copy = batch.shutil.copy

    def _truncating_copy(src, dst):
        real_copy(src, dst)
        if dst.endswith("ntuple.1.0.root"):
            with open(dst, "wb") as f:      # a short write at the destination
                f.write(b"ro")

    monkeypatch.setattr(batch.shutil, "copy", _truncating_copy)

    res = _run_publish(batch, scan, tar, eos)

    assert res.exit_code != 0
    assert "refusing to delete" in res.output
    assert scan.is_dir(), "local scan dir must survive an unverified publish"
    assert tar.is_file(), "local tarball must survive an unverified publish"


def test_publish_cleanup_keeps_the_bookkeeping_outside_the_scan_dir(batch, tmp_path, monkeypatch):
    '''r3mg/ is what makes the run resumable, and it is never inside the scan
    dir — a sibling directory must not be touched by the cleanup.'''
    scan, tar = _make_batch_dir(tmp_path)
    meta = tmp_path / "r3mg" / "scan.1.0"
    meta.mkdir(parents=True)
    (meta / "step_SPheno.json").write_text("{}")
    eos = tmp_path / "eos"
    monkeypatch.setenv("EOSPATH", str(eos))

    _run_publish(batch, scan, tar, eos)

    assert (meta / "step_SPheno.json").is_file()


# ── the marker must land BEFORE the local delete ───────────────────────────

def _run_publish_with_marker(batch, scan, tar, eospath, marker, cleanup=True):
    args = ["--scan_dir", str(scan), "--tarball", str(tar), "--runid", "1",
            "--procid", "0", "--marker", str(marker)]
    if cleanup:
        args.append("--cleanup")
    return CliRunner().invoke(batch.cli, ["publish"] + args,
                              env={"EOSPATH": str(eospath)})


def test_marker_is_written_before_the_local_copies_are_deleted(batch, tmp_path, monkeypatch):
    '''The stranding window: a job killed after the delete but before the marker
    leaves a batch that resume can neither rebuild (inputs gone) nor skip (no
    marker). Ordering is the whole fix, so assert the order, not just the result.'''
    scan, tar = _make_batch_dir(tmp_path)
    eos = tmp_path / "eos"
    marker = tmp_path / "r3mg" / "scan.1.0" / "published.ok"
    monkeypatch.setenv("EOSPATH", str(eos))

    seen = {}
    real_rmtree = batch.shutil.rmtree

    def _spy(path, *a, **k):
        seen["marker_existed_at_delete"] = marker.is_file()
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(batch.shutil, "rmtree", _spy)

    res = _run_publish_with_marker(batch, scan, tar, eos, marker)

    assert res.exit_code == 0, res.output
    assert seen.get("marker_existed_at_delete") is True, \
        "marker must exist before the scan dir is removed"
    assert marker.is_file() and not scan.exists()


def test_no_marker_when_the_published_copies_do_not_verify(batch, tmp_path, monkeypatch):
    '''Failing verification must leave neither a marker nor a deleted scan dir,
    so the batch simply re-publishes next attempt.'''
    scan, tar = _make_batch_dir(tmp_path)
    eos = tmp_path / "eos"
    marker = tmp_path / "r3mg" / "scan.1.0" / "published.ok"
    monkeypatch.setenv("EOSPATH", str(eos))

    real_copy = batch.shutil.copy

    def _truncating_copy(src, dst):
        real_copy(src, dst)
        if dst.endswith("ntuple.1.0.root"):
            with open(dst, "wb") as f:
                f.write(b"ro")

    monkeypatch.setattr(batch.shutil, "copy", _truncating_copy)

    res = _run_publish_with_marker(batch, scan, tar, eos, marker)

    assert res.exit_code != 0
    assert not marker.exists(), "an unverified publish must not record success"
    assert scan.is_dir() and tar.is_file()


def test_marker_is_written_without_cleanup_too(batch, tmp_path, monkeypatch):
    '''--keep_local still needs the marker: it is the rule's declared output.'''
    scan, tar = _make_batch_dir(tmp_path)
    eos = tmp_path / "eos"
    marker = tmp_path / "r3mg" / "scan.1.0" / "published.ok"
    monkeypatch.setenv("EOSPATH", str(eos))

    res = _run_publish_with_marker(batch, scan, tar, eos, marker, cleanup=False)

    assert res.exit_code == 0, res.output
    assert marker.is_file() and scan.is_dir() and tar.is_file()
