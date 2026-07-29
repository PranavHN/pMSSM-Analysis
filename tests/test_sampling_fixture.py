'''Golden-file check: the sampled parameter points must not drift.

The CI regression check compares *per-step success counts* against references on
EOS ("2 of 20 models survived SPheno"). Those counts only mean anything if the
20 models are the same 20 models as when the references were produced — and the
points come from a single np.random stream, so reordering draws inside a
sampler, or a numpy version bump, silently changes every one of them. Without
this guard such a change surfaces as a reference mismatch that looks like a
physics regression instead of what it is.

The fixture was frozen from the pre-refactor code (origin/main @ 4218c3a), so it
also documents that extracting sampling.py out of ModelGenerator changed nothing.

Deliberately driven through ModelGenerator rather than sampling.py directly:
that is the path a real run takes (config merge -> np.random.seed in __init__ ->
sampler delegation), and all of it has to stay stable, not just the samplers.

Regenerating after an intentional sampling change: see
tests/regen_sampling_fixture.py — the EOS references must be re-seeded in the
same commit.
'''

import json
from pathlib import Path

import pytest

from Run3ModelGen.modelgen import ModelGenerator

FIXTURE = Path(__file__).parent / "fixtures" / "sampling_seed123.json"


def pytest_generate_tests(metafunc):
    '''Parametrise over the frozen cases so a failure names the case.'''
    if "case_name" in metafunc.fixturenames:
        with open(FIXTURE) as f:
            metafunc.parametrize("case_name", sorted(json.load(f)))


@pytest.fixture(scope="module")
def fixture_data():
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture
def sampled(case_name, fixture_data, datadir, tmp_path, monkeypatch):
    '''Replay one frozen case and return (generator, expected).'''
    case = fixture_data[case_name]
    config_file = case["config_file"]
    if config_file is not None:
        config_file = str(datadir / "configs" / config_file)

    # The importance configs name their seed CSV by bare filename, resolved
    # against the cwd (as in the run/ directory CI uses).
    for csv in ("ci_importance_seeds.csv", "example_importance_seeds.csv"):
        (tmp_path / csv).write_bytes((datadir / "configs" / csv).read_bytes())
    monkeypatch.chdir(tmp_path)

    mg = ModelGenerator(config_file=config_file,
                        scan_dir=f"scan_{case_name}", seed=case["seed"], quiet=True)
    getattr(mg, case["sampler"])()
    return mg, case


def test_num_models_is_unchanged(sampled, case_name):
    mg, case = sampled
    assert int(mg.num_models) == case["num_models"], (
        f"{case_name}: sampler produced {mg.num_models} models, fixture says {case['num_models']}")


def test_every_sampled_point_is_bit_identical(sampled, case_name):
    mg, case = sampled
    assert set(mg.points) >= set(case["points"]), f"{case_name}: parameters disappeared from the sampler"
    for key, expected in case["points"].items():
        got = [float(v) for v in mg.points[key]]
        if got == expected:
            continue
        ndiff = sum(1 for a, b in zip(got, expected) if a != b) + abs(len(got) - len(expected))
        first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        detail = "" if first is None else f"; first at index {first}: {got[first]!r} != {expected[first]!r}"
        pytest.fail(f"{case_name}: points['{key}'] drifted in {ndiff} of {len(expected)} entries{detail}. "
                    f"If the change is intentional, regenerate with "
                    f"tests/regen_sampling_fixture.py and re-seed the EOS references.")


def test_the_fixture_covers_every_point_sampled_prior(fixture_data):
    '''A prior with no frozen case would drift unnoticed.'''
    samplers = {c["sampler"] for c in fixture_data.values()}
    assert samplers == {"sample_flat", "sample_fixed", "sample_grid", "sample_importance"}


def test_the_fixture_pins_more_than_one_seed(fixture_data):
    '''A single seed could match by accident after a stream change.'''
    assert len({c["seed"] for c in fixture_data.values()}) > 1
