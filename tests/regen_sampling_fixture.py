#!/usr/bin/env python
'''Regenerate the sampled-point golden file used by test_sampling_fixture.py.

Only run this when sampling has changed ON PURPOSE (a new prior, a deliberate
change to the draw order, a numpy pin bump). Regenerating it to make a red test
go green otherwise would erase the guard: every scan would produce different
parameter points than the EOS references were built from.

    pixi run python tests/regen_sampling_fixture.py            # rewrite the fixture
    pixi run python tests/regen_sampling_fixture.py --show     # print, change nothing

The full procedure for an intentional change — all in ONE commit, so the two
contracts can never disagree:

  1. regenerate the fixture with this script and eyeball the diff (a changed
     draw order rewrites every value; a new prior only adds a case),
  2. `pixi run make-refs` and review the new per-step success counts,
  3. `pixi run upload-refs` to re-seed the EOS references,
  4. commit fixture + any config/sampler change together.

Adding a case: extend CASES below (a config in data/configs/ plus the sampler
method to call) and rerun. test_sampling_fixture.py asserts that every
point-sampled prior is covered and that more than one seed is pinned.
'''

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "source" / "Run3ModelGen" / "data"
FIXTURE = Path(__file__).parent / "fixtures" / "sampling_seed123.json"

# Importable without a built environment, exactly like tests/conftest.py.
sys.path.insert(0, str(REPO / "build"))
os.environ.setdefault("DATAPATH", str(DATA))

# name -> (config file in data/configs or None for the default, seed, sampler)
CASES = [
    ("flat_default_123",    None,                              123,  "sample_flat"),
    ("flat_default_4242",   None,                              4242, "sample_flat"),
    ("fixed_example_123",   "example_fixed_config.yaml",       123,  "sample_fixed"),
    ("grid_ci_123",         "ci_grid_config.yaml",             123,  "sample_grid"),
    ("grid_example_123",    "example_grid_config.yaml",        123,  "sample_grid"),
    ("importance_ci_123",   "ci_importance_config.yaml",       123,  "sample_importance"),
    ("importance_ex_123",   "example_importance_config.yaml",  123,  "sample_importance"),
]


def build() -> dict:
    from Run3ModelGen.modelgen import ModelGenerator

    # Seed CSVs are named by bare filename in the importance configs.
    workdir = tempfile.mkdtemp(prefix="regen_sampling_")
    for csv in ("ci_importance_seeds.csv", "example_importance_seeds.csv"):
        shutil.copy(DATA / "configs" / csv, workdir)
    os.chdir(workdir)

    fixture = {}
    for name, config, seed, sampler in CASES:
        config_file = str(DATA / "configs" / config) if config else None
        mg = ModelGenerator(config_file=config_file, scan_dir=f"scan_{name}",
                            seed=seed, quiet=True)
        getattr(mg, sampler)()
        fixture[name] = {
            "config_file": config,
            "seed": seed,
            "sampler": sampler,
            "num_models": int(mg.num_models),
            "points": {k: [float(x) for x in v] for k, v in mg.points.items()},
        }
        print(f"  {name:22} {mg.num_models:>5} models", file=sys.stderr)
    return fixture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="print the fixture instead of writing it")
    args = ap.parse_args()

    print(f"Sampling {len(CASES)} cases...", file=sys.stderr)
    # The samplers log to stdout; keep it clean so `--show` is pipeable
    # (the logging config is captured on import, which happens inside build()).
    with contextlib.redirect_stdout(sys.stderr):
        fixture = build()
    text = json.dumps(fixture, indent=1, sort_keys=True)

    if args.show:
        print(text)
        return 0

    FIXTURE.write_text(text)
    print(f"\nWrote {FIXTURE}\nReview the diff, then re-seed the EOS references "
          f"(pixi run make-refs && pixi run upload-refs) in the same commit.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
