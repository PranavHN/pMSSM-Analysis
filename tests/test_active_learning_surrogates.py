'''The torch half of prior: active_learning, on the CPU: every surrogate x head x
strategy combination the config allows, on tiny synthetic problems, plus the loop end to
end with a prep_input-only pipeline (the target is an input branch, so no physics tool is
needed and the surrogate can learn it).

Skipped in the default environment (no torch). The GPU refusal lives in the rule body
(``require_gpu``), not in the models, which is what lets this run on a CPU.'''
import json
import os

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from Run3ModelGen import pipeline
from Run3ModelGen.active_learning import acquisition, config as alc, data as ald
from Run3ModelGen.active_learning.heads import get_head
from Run3ModelGen.active_learning.models import make_surrogate

DEVICE = "cpu"


def _cfg(default_config, **block):
    merged = pipeline.load_scan_config(None, default_config)
    merged["prior"] = "active_learning"
    merged["num_models"] = 8
    base = {"training": {"epochs": 30, "patience": 10}, "acquisition": {"mc_samples": 6, "entropy_pool_size": 40},
            "candidates": {"n_candidates": 200}, "gp": {"num_inducing_max": 16}}
    merged["active_learning"] = alc.deep_merge(base, block)
    return alc.load_al_config(merged)


def _toy(n=120, seed=0):
    '''t = x0 + 0.5 x1 (so the boundary t = 0 is a line); X in [-1, 1]^2.'''
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    t = X[:, 0] + 0.5 * X[:, 1]
    return X, t


COMBOS = [(m, h, s) for m in alc.MODELS for h, ss in alc.STRATEGIES_BY_HEAD.items() for s in ss]


@pytest.mark.parametrize("model,head,strategy", COMBOS, ids=[f"{m}-{h}-{s}" for m, h, s in COMBOS])
def test_every_combination_trains_evaluates_and_selects(default_config, tmp_path, model, head, strategy):
    cfg = _cfg(default_config, model=model, head=head, strategy=strategy)
    X, t = _toy()
    tr, va = np.arange(90), np.arange(90, 120)
    sur = make_surrogate(cfg, 2, DEVICE, lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), seed=3)
    log_path = tmp_path / "training.log"
    metrics = sur.fit(X[tr], t[tr], X[va], t[va], log_path=str(log_path))
    assert log_path.exists() and log_path.stat().st_size > 0
    assert metrics["n_train"] == 90 and metrics["epochs_run"] >= 1
    assert 0.0 <= metrics["verdict_accuracy_val"] <= 1.0
    ckpt = sur.checkpoint()
    assert "model_state_dict" in ckpt

    cand = ald.generate_candidates({"a": [-1, 1], "b": [-1, 1]}, 200, "lhs", seed=1)
    ev = sur.evaluate(cand)
    assert ev.mean.shape == (200,) and ev.var.shape == (200,) and ev.summary["score"].shape == (200,)
    assert torch.isfinite(ev.mean).all() and torch.isfinite(ev.summary["score"]).all()
    sel = acquisition.select(strategy, sur, ev, 8, cfg.acquisition, cfg.threshold, 11, DEVICE)
    assert 1 <= len(sel.indices) <= 8 and len(set(sel.indices.tolist())) == len(sel.indices)
    assert sel.pred.shape == sel.indices.shape and sel.score.shape == sel.indices.shape

    # warm start: the previous weights load (shape-compatible keys for the GPs)
    sur2 = make_surrogate(cfg, 2, DEVICE, lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), seed=4)
    m2 = sur2.fit(X[tr], t[tr], X[va], t[va], warm=ckpt)
    assert m2["epochs_run"] >= 1


def test_cls_entropy_peaks_on_the_decision_boundary(default_config):
    cfg = _cfg(default_config, model="dnn", head="classification", strategy="cls_entropy",
               training={"epochs": 200, "patience": 50})
    X, t = _toy(400)
    sur = make_surrogate(cfg, 2, DEVICE, lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), seed=3)
    sur.fit(X[:320], t[:320], X[320:], t[320:])
    cand = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0], [0.5, -1.0]])   # t = 1.5, -1.5, 0, 0
    ev = sur.evaluate(cand)
    s = ev.summary["score"].numpy()
    assert s[2] > s[0] and s[2] > s[1], "entropy is largest at the boundary"
    assert ev.summary["p_mean"][0] > 0.5 > ev.summary["p_mean"][1]


def test_tol_only_random_cut_and_seed(default_config):
    cfg = _cfg(default_config, model="dnn", head="regression", strategy="tol_only_random",
               acquisition={"tolerance": 0.5})
    X, t = _toy(200)
    sur = make_surrogate(cfg, 2, DEVICE, lo=np.array([-1.0, -1.0]), hi=np.array([1.0, 1.0]), seed=3)
    sur.fit(X[:160], t[:160], X[160:], t[160:])
    cand = ald.generate_candidates({"a": [-1, 1], "b": [-1, 1]}, 300, "uniform", seed=2)
    ev = sur.evaluate(cand)
    a = acquisition.select("tol_only_random", sur, ev, 20, cfg.acquisition, 0.0, 7, DEVICE)
    b = acquisition.select("tol_only_random", sur, ev, 20, cfg.acquisition, 0.0, 7, DEVICE)
    assert np.array_equal(a.indices, b.indices), "seeded"
    assert (np.abs(a.pred) < 0.5).all(), "only survivors of the cut are drawn"
    assert np.isnan(a.score).all()


def test_heads_summaries():
    reg, cls = get_head("regression"), get_head("classification", link="logistic")
    samples = torch.tensor([[0.0, 2.0, -2.0], [0.2, 2.2, -1.8], [-0.2, 1.8, -2.2]])
    r = reg.summarise(samples)
    assert torch.allclose(r["mean"], torch.tensor([0.0, 2.0, -2.0])) and torch.equal(r["score"], r["var"])
    c = cls.summarise(samples)
    assert c["score"][0] > c["score"][1] and c["score"][0] > c["score"][2]
    g = get_head("classification", link="probit").summarise_gaussian(torch.tensor([0.0, 3.0]), torch.tensor([1.0, 0.1]))
    assert g["p_mean"][0] == pytest.approx(0.5, abs=1e-6) and g["p_mean"][1] > 0.99
    lsq = get_head("lsq_classification")
    assert lsq.make_targets(torch.tensor([-0.5, 0.5])).tolist() == [-1.0, 1.0]


# ── the loop, end to end, with a prep_input-only pipeline ───────────────────────

PARAMS = {"tanb": [1, 60], "M_1": [-2000, 2000], "M_2": [-2000, 2000], "M_3": [4000, 4000],
          "AT": [-8000, 8000], "Ab": [-2000, 2000], "Atau": [-2000, 2000], "mu": [-2000, 2000],
          "mA": [2000, 2000], "meL": [0, 2000], "mtauL": [2000, 2000], "meR": [0, 2000],
          "mtauR": [2000, 2000], "mqL1": [4000, 4000], "mqL3": [4000, 4000], "muR": [4000, 4000],
          "mtR": [4000, 4000], "mdR": [4000, 4000], "mbR": [4000, 4000]}


@pytest.fixture
def loop_config(tmp_path):
    cfg = {"prior": "active_learning", "num_models": 6, "isGMSB": False, "parameters": PARAMS,
           "steps": [{"name": "prep_input", "output_dir": "input", "prefix": "IN"}],
           "active_learning": {
               "environment": "rocm", "nworkers": 2, "seed_models": 40, "n_iterations": 2,
               "model": "dnn", "head": "classification", "strategy": "cls_entropy",
               "candidates": {"n_candidates": 300}, "training": {"epochs": 20, "patience": 10},
               "acquisition": {"mc_samples": 4},
               "labelling": {"min_valid_fraction": 0.5, "max_attempts": 2},
               "target": {"branch": "IN_tanb", "value": 30.0, "transform": "log",
                          "valid": {"lo": 5.0, "hi": None}, "hit_tolerance": 0.10},
               "vetoes": [{"branch": "IN_M_1", "lo": -1500.0}]}}
    path = tmp_path / "al.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_loop_end_to_end_and_resume(loop_config, tmp_path):
    from Run3ModelGen.active_learning import loop
    scan = tmp_path / "scan"
    summary = loop.run_active_learning_scan(loop_config, str(scan), seed=5, nworkers=2, max_hours=1,
                                            worker_base=str(tmp_path / "w"), device=DEVICE)
    assert summary["complete"] and summary["iterations_done"] == 2
    for it in range(3):
        d = scan / f"iteration_{it:03d}"
        assert (d / "ntuple.0.0.root").exists() and (d / "scan_config.yaml").exists()
        assert (d / "selected_points.csv").exists() and (d / loop.ITERATION_JSON).exists()
        assert (d / "input").is_dir()
        if it:
            assert (d / "training.log").stat().st_size > 0
            assert (d / "model.pt").exists() and (d / "metrics.json").exists()
    assert (scan / "ntuple.0.0.root").exists() and (scan / loop.STATE_FILE).exists()
    assert (scan / loop.MAIN_LOG).stat().st_size > 0
    assert (scan / loop.SUMMARY_JSON).exists()
    rec = json.load(open(scan / "iteration_001" / loop.ITERATION_JSON))
    assert rec["n_labelled"] >= 6 and rec["training"]["n_train"] > 0

    import uproot
    with uproot.open(str(scan / "ntuple.0.0.root")) as f:
        t = f["susy"]
        n0 = 40
        assert t.num_entries == n0 + rec["n_labelled"] + json.load(open(scan / "iteration_002" / loop.ITERATION_JSON))["n_labelled"]
        its = t["al_iteration"].array(library="np")
        assert (its[:n0] == 0).all() and its.max() == 2
        assert (t["al_pred"].array(library="np")[:n0] == -1).all()

    # resume: a third iteration is added, the first two are untouched
    with open(loop_config) as f:
        cfg = yaml.safe_load(f)
    cfg["active_learning"]["n_iterations"] = 3
    with open(loop_config, "w") as f:
        yaml.safe_dump(cfg, f)
    before = (scan / "iteration_001" / "ntuple.0.0.root").stat().st_mtime
    summary2 = loop.run_active_learning_scan(loop_config, str(scan), seed=5, nworkers=2, max_hours=1,
                                             worker_base=str(tmp_path / "w2"), device=DEVICE)
    assert summary2["complete"] and summary2["iterations_done"] == 3
    assert (scan / "iteration_003" / loop.ITERATION_JSON).exists()
    assert (scan / "iteration_001" / "ntuple.0.0.root").stat().st_mtime == before
    assert len(summary2["iterations"]) == 4


def test_wall_clock_guard_checkpoints(loop_config, tmp_path):
    from Run3ModelGen.active_learning import loop
    scan = tmp_path / "scan_guard"
    summary = loop.run_active_learning_scan(loop_config, str(scan), seed=5, nworkers=2,
                                            max_hours=1e-9, worker_base=str(tmp_path / "w"), device=DEVICE)
    assert not summary["complete"] and "wall-clock" in summary["stopped"]
    assert not (scan / "ntuple.0.0.root").exists()
