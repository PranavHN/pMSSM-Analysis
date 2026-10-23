'''Labelling a batch into an iteration scan dir, and merging iteration ntuples.

Uses a prep_input-only pipeline, which needs no physics binary: the ntuple then carries
the IN_* branches, and a target defined on one of them (IN_tanb) exercises the whole
label/veto/hit path end to end, worker pool included. Torch-free.'''
import json
import os

import numpy as np
import pytest
import yaml

from Run3ModelGen import pipeline
from Run3ModelGen.active_learning import data as ald
from Run3ModelGen.active_learning import labelling, ntuple as alnt
from Run3ModelGen.active_learning.config import TargetSpec, Veto

PARAMS = {"tanb": [1, 60], "M_1": [-2000, 2000], "M_2": [-2000, 2000], "M_3": [4000, 4000],
          "AT": [-8000, 8000], "Ab": [-2000, 2000], "Atau": [-2000, 2000], "mu": [-2000, 2000],
          "mA": [2000, 2000], "meL": [0, 2000], "mtauL": [2000, 2000], "meR": [0, 2000],
          "mtauR": [2000, 2000], "mqL1": [4000, 4000], "mqL3": [4000, 4000], "muR": [4000, 4000],
          "mtR": [4000, 4000], "mdR": [4000, 4000], "mbR": [4000, 4000]}
TARGET = TargetSpec(branch="IN_tanb", value=30.0, transform="log", valid_lo=5.0, valid_hi=None,
                    hit_tolerance=0.10)
VETOES = (Veto(branch="IN_M_1", lo=-1500.0),)


@pytest.fixture(scope="module")
def config_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfg")
    cfg = {"prior": "active_learning", "num_models": 4, "isGMSB": False, "parameters": PARAMS,
           "steps": [{"name": "prep_input", "output_dir": "input", "prefix": "IN"}],
           "active_learning": {"model": "dnn", "nworkers": 2}}
    path = d / "al.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.fixture(scope="module")
def pool(config_file, tmp_path_factory):
    scratch = tmp_path_factory.mktemp("scratch")
    p = labelling.start_pool(2, config_file, str(scratch), seed=5)
    yield p
    p.close()
    p.join()


def test_label_batch_builds_an_iteration_scan_dir(pool, config_file, tmp_path):
    it_dir = tmp_path / "iteration_000"
    X = np.array([[40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0],   # tanb 40: valid, not a hit
                  [31.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0],  # tanb 31: valid hit
                  [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0],    # tanb 3: below valid_lo
                  [29.0, -1800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0]])  # vetoed by M_1
    free = ald.free_parameters(PARAMS)
    assert free == ["tanb", "M_1", "M_2", "AT", "Ab", "Atau", "mu", "meL", "meR"]
    points = ald.to_points(X, PARAMS, free)
    batch = labelling.label_batch(pool, config_file, str(it_dir), points, 0, 0, 5, TARGET, VETOES)

    assert list(batch.indices) == [0, 1, 2, 3]
    assert list(batch.valid) == [True, True, False, False]
    assert list(batch.hit) == [False, True, False, False]
    assert batch.y == pytest.approx([40.0, 31.0, 3.0, 29.0])
    assert batch.counts["prep_input"] == [4, 4, 4]
    # the ordinary scan-dir layout
    assert sorted(os.listdir(it_dir / "input")) == ["0.slha", "1.slha", "2.slha", "3.slha"]
    assert (it_dir / "ntuple.0.0.root").exists()
    assert (it_dir / "scan_config.yaml").exists()
    steps = json.load(open(it_dir / labelling.STEPS_JSON))
    assert steps["num_models"] == 4 and steps["attempts"][0]["steps"]["prep_input"]["counts"] == [4, 4]
    recorded = yaml.safe_load(open(it_dir / "scan_config.yaml"))
    assert recorded["prior"] == "active_learning" and recorded["num_models"] == 4

    # a second attempt appends with continuing indices and one ntuple holds both
    X2 = np.array([[33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0]])
    batch2 = labelling.label_batch(pool, config_file, str(it_dir), ald.to_points(X2, PARAMS, free),
                                   4, 1, 5, TARGET, VETOES)
    assert list(batch2.indices) == [4] and list(batch2.hit) == [False] and list(batch2.valid) == [True]
    import uproot
    with uproot.open(str(it_dir / "ntuple.0.0.root")) as f:
        tree = f["susy"]
        assert tree.num_entries == 5
        assert tree["model"].array(library="np").astype(int).tolist() == [0, 1, 2, 3, 4]
        assert tree["IN_tanb"].array(library="np") == pytest.approx([40, 31, 3, 29, 33])
    steps = json.load(open(it_dir / labelling.STEPS_JSON))
    assert [a["attempt"] for a in steps["attempts"]] == [0, 1] and steps["num_models"] == 5


def test_read_labels_handles_a_missing_branch(tmp_path):
    import uproot
    path = tmp_path / "ntuple.0.0.root"
    with uproot.recreate(str(path)) as f:
        f["susy"] = {"model": np.arange(3.0), "IN_tanb": np.array([10.0, 20.0, 30.0])}
    y, valid, hit = labelling.read_labels(str(path), TargetSpec("MO_Omega", 0.12), (), [0, 1, 2])
    assert list(y) == [-1.0, -1.0, -1.0] and not valid.any() and not hit.any()
    with pytest.raises(RuntimeError, match="out of sync"):
        labelling.read_labels(str(path), TARGET, (), [5])


def test_merge_iterations_keeps_the_batch_contract(tmp_path):
    import uproot
    scan = tmp_path / "scan"
    for it, (tanb, omega) in enumerate([([10.0, 40.0], [0.5, 0.12]), ([31.0], [0.125])]):
        d = scan / f"iteration_{it:03d}"
        d.mkdir(parents=True)
        tree = {"model": np.arange(float(len(tanb))), "IN_tanb": np.array(tanb), "IN_M_1": np.zeros(len(tanb))}
        if it == 1:
            tree["MO_Omega"] = np.array(omega)     # a branch only the second iteration has
        with uproot.recreate(str(d / "ntuple.0.0.root")) as f:
            f["susy"] = tree
        if it == 1:
            json.dump({"strategy": "cls_entropy", "attempts": [],
                       "models": [{"index": 0, "attempt": 0, "pred": 0.3, "score": 0.69}]},
                      open(d / alnt.SELECTION_JSON, "w"))
    out = tmp_path / "ntuple.7.3.root"
    n, nb = alnt.merge_iterations(str(scan), str(out), TARGET, VETOES, batch_index=3)
    assert n == 3
    with uproot.open(str(out)) as f:
        t = f["susy"]
        assert set(alnt.AL_BRANCHES) <= set(t.keys())
        assert t["model"].array(library="np").astype(int).tolist() == [300000, 300001, 300002]
        assert t["al_iteration"].array(library="np").tolist() == [0, 0, 1]
        assert t["al_index"].array(library="np").tolist() == [0, 1, 0]
        assert t["MO_Omega"].array(library="np").tolist() == [-1.0, -1.0, 0.125], "-1 fill where absent"
        assert t["al_valid"].array(library="np").tolist() == [1.0, 1.0, 1.0]
        assert t["al_hit"].array(library="np").tolist() == [0.0, 0.0, 1.0]
        assert t["al_pred"].array(library="np").tolist() == [-1.0, -1.0, pytest.approx(0.3)]
        assert t["al_score"].array(library="np").tolist() == [-1.0, -1.0, pytest.approx(0.69)]
    assert alnt.iteration_dirs(str(scan)) == [str(scan / "iteration_000"), str(scan / "iteration_001")]
