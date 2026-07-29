'''Shared fixtures for the Run3ModelGen unit tests.

The tests cover the *pure* layer only — config/sampling/pipeline/facade logic
and the tool wrappers driven by stub executables — so they need no physics
software, no cluster and no network, and run in seconds. End-to-end coverage
(real tools, reference counts, batch contracts) lives in CI.
'''

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG_PARENT = REPO / "source" / "Run3ModelGen"

# Some modules (modelgen, extractor) resolve DATAPATH at import time. Point it at
# this checkout unless the environment already sets it, so the suite runs both
# standalone and after `. build/setup.sh`. Set before any test module imports
# them — never skipped: a silently skipped contract guard is worse than none.
os.environ.setdefault("DATAPATH", str(PKG_PARENT / "data"))

# Importable without `source build/setup.sh`: expose the python/ dir under the
# package name the modules use for their absolute imports.
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))
    _pkg_link = PKG_PARENT / "Run3ModelGen"
    if not _pkg_link.exists():
        # No writable link available (read-only checkout): fall back to the build
        # tree that cmake_build_env.sh populates.
        build = REPO / "build"
        if (build / "Run3ModelGen").exists():
            sys.path.insert(0, str(build))


@pytest.fixture(scope="session")
def repo():
    return REPO


@pytest.fixture(scope="session")
def datadir(repo):
    return repo / "source" / "Run3ModelGen" / "data"


@pytest.fixture(scope="session")
def default_config(datadir):
    return str(datadir / "configs" / "default_config.yaml")


@pytest.fixture(scope="session")
def raw_slha(datadir):
    return str(datadir / "raw.slha")


@pytest.fixture
def stub_bin(tmp_path, monkeypatch):
    '''Factory putting a stub executable at the front of PATH.

    Lets the tool wrappers be tested for their file/cwd handling (the part that
    is our code) without the real physics binaries.
    '''
    bindir = tmp_path / "stub_bin"
    bindir.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")

    def _make(name, script):
        path = bindir / name
        path.write_text("#!/bin/bash\n" + script)
        path.chmod(0o755)
        return path

    return _make
