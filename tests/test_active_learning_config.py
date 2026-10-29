'''The active_learning config block: defaults, deep merge, validation, target semantics.

Torch-free (the block is validated by ModelGenerator in the default environment).'''
import copy

import numpy as np
import pytest
import yaml

from Run3ModelGen import pipeline
from Run3ModelGen.active_learning import config as alc


@pytest.fixture
def merged(default_config):
    cfg = pipeline.load_scan_config(None, default_config)
    cfg["prior"] = "active_learning"
    return cfg


def _with(merged, **block):
    m = copy.deepcopy(merged)
    m["active_learning"] = alc.deep_merge(m.get("active_learning") or {}, block)
    return m


def test_defaults_match_the_default_config_yaml(default_config):
    '''config.DEFAULTS and the block in default_config.yaml are the same dict: the
    deep merge works from DEFAULTS, the documentation from the yaml.'''
    with open(default_config) as f:
        block = yaml.safe_load(f)["active_learning"]
    assert block == alc.DEFAULTS


def test_the_default_block_validates(merged):
    cfg = alc.load_al_config(merged)
    assert cfg.model == "transformer" and cfg.head == "classification" and cfg.strategy == "cls_entropy"
    assert cfg.environment == "rocm"
    assert cfg.num_models == merged["num_models"]
    assert cfg.target.branch == "MO_Omega" and cfg.target.threshold == 0.0
    assert [v.branch for v in cfg.vetoes] == ["SP_m_h", "SP_LSP_type"]


def test_partial_user_block_keeps_the_other_defaults(merged):
    cfg = alc.load_al_config(_with(merged, model="dnn", training={"epochs": 7}))
    assert cfg.model == "dnn"
    assert cfg.training["epochs"] == 7
    assert cfg.training["patience"] == alc.DEFAULTS["training"]["patience"]
    assert cfg.acquisition == alc.DEFAULTS["acquisition"]


def test_lists_replace_rather_than_merge(merged):
    cfg = alc.load_al_config(_with(merged, vetoes=[{"branch": "SP_m_h", "lo": 0}]))
    assert [v.branch for v in cfg.vetoes] == ["SP_m_h"]


@pytest.mark.parametrize("head,strategy", [
    ("classification", "entropy_batch"), ("classification", "tol_only_random"),
    ("regression", "cls_entropy"), ("regression", "bald"), ("classification", "top_k"),
])
def test_head_strategy_mismatch_is_refused(merged, head, strategy):
    with pytest.raises(alc.AlConfigError, match="strategy"):
        alc.load_al_config(_with(merged, head=head, strategy=strategy))


@pytest.mark.parametrize("head,strategy", [
    ("classification", "cls_entropy"), ("regression", "entropy_batch"), ("regression", "tol_only_random"),
])
@pytest.mark.parametrize("model", alc.MODELS)
def test_every_supported_combination_validates(merged, model, head, strategy):
    cfg = alc.load_al_config(_with(merged, model=model, head=head, strategy=strategy))
    assert (cfg.model, cfg.head, cfg.strategy) == (model, head, strategy)


def test_the_default_environment_is_refused(merged):
    '''No CPU mode: environment: default must fail at config load, with the hint.'''
    with pytest.raises(alc.AlConfigError, match="no CPU mode"):
        alc.load_al_config(_with(merged, environment="default"))


def test_unknown_environment_is_refused(merged):
    with pytest.raises(alc.AlConfigError, match="environment"):
        alc.load_al_config(_with(merged, environment="metal"))


@pytest.mark.parametrize("block,key", [
    ({"target": {"value": -1.0}}, "target.value"),
    ({"target": {"value": 0.0, "transform": "none"}}, "target.value"),
    ({"target": {"valid": {"lo": 2.0, "hi": 1.0}}}, "target.valid"),
    ({"target": {"hit_tolerance": 0}}, "target.hit_tolerance"),
    ({"target": {"branch": ""}}, "target.branch"),
    ({"target": {"transform": "log10"}}, "target.transform"),
    ({"vetoes": [{"branch": "SP_m_h"}]}, "vetoes"),
    ({"vetoes": [{"branch": "SP_m_h", "lo": 3, "hi": 1}]}, "vetoes"),
    ({"nworkers": 0}, "nworkers"),
    ({"seed_models": 0}, "seed_models"),
    ({"seed_models": 20000}, "seed_models"),
    ({"val_fraction": 1.0}, "val_fraction"),
    ({"candidates": {"generation": "sobol"}}, "candidates.generation"),
    ({"training": {"epochs": 0}}, "training.epochs"),
    ({"transformer": {"d_model": 100, "nhead": 3}}, "transformer"),
    ({"transformer": {"dropout": 0.0}}, "transformer.dropout"),
    ({"gp": {"kernel": "Linear"}}, "gp.kernel"),
    ({"acquisition": {"mc_samples": 1}}, "acquisition.mc_samples"),
    ({"labelling": {"min_valid_fraction": 0}}, "labelling.min_valid_fraction"),
    ({"labelling": {"max_attempts": 0}}, "labelling.max_attempts"),
    ({"model": "tabpfn"}, "model"),
])
def test_bad_keys_are_named(merged, block, key):
    with pytest.raises(alc.AlConfigError, match=key.replace(".", r"\.").replace("[", r"\[")):
        alc.load_al_config(_with(merged, **block))


def test_num_models_cap_is_enforced(merged):
    m = copy.deepcopy(merged)
    m["num_models"] = pipeline.MAX_MODELS_PER_JOB + 1
    with pytest.raises(alc.AlConfigError, match="num_models"):
        alc.load_al_config(m)


def test_zero_dropout_is_fine_for_the_gps(merged):
    cfg = alc.load_al_config(_with(merged, model="exact_gp", head="regression",
                                   strategy="entropy_batch", transformer={"dropout": 0.0}))
    assert cfg.model == "exact_gp"


# ── target semantics ───────────────────────────────────────────────────────────

def test_log_target_puts_the_value_at_zero():
    t = alc.TargetSpec(branch="MO_Omega", value=0.12, transform="log", valid_lo=0.0, valid_hi=1.0)
    y = np.array([0.12, 0.24, 0.06, -1.0, 2.0, 0.0])
    tt = t.transform_y(y)
    assert tt[0] == pytest.approx(0.0) and tt[1] == pytest.approx(np.log(2)) and tt[2] == pytest.approx(-np.log(2))
    assert t.inverse_transform_y(tt[:3]) == pytest.approx(y[:3])
    assert t.threshold == 0.0
    assert list(t.is_valid(y)) == [True, True, True, False, False, False]


def test_sentinel_never_valid_and_hits_are_relative():
    t = alc.TargetSpec(branch="MO_Omega", value=0.12, transform="log", hit_tolerance=0.10)
    y = np.array([-1.0, 0.12, 0.131, 0.133, 0.109, 0.107, np.nan])
    assert list(t.is_valid(y)) == [False, True, True, True, True, True, False]
    assert list(t.is_hit(y)) == [False, True, True, False, True, False, False]


def test_none_transform_keeps_the_value_as_threshold():
    t = alc.TargetSpec(branch="X", value=0.05, transform="none", valid_lo=None, valid_hi=None)
    assert t.threshold == 0.05
    assert t.transform_y(np.array([0.3]))[0] == 0.3
    assert list(t.is_valid(np.array([-1.0, 0.3]))) == [True, True]   # no window: -1 counts as a value


def test_veto_is_a_strict_window():
    v = alc.Veto(branch="SP_LSP_type", lo=0.5, hi=3.5)
    assert list(v.passes(np.array([-1, 1, 2, 3, 4, 0.5, 3.5]))) == [False, True, True, True, False, False, False]
    assert list(alc.Veto("SP_m_h", lo=0.0).passes(np.array([-1.0, 125.0]))) == [False, True]


def test_modelgenerator_validates_the_block(tmp_path, default_config):
    '''A bad block must fail at ModelGenerator construction (config load), like a bad
    constraints block does for prior: mcmc.'''
    from Run3ModelGen.modelgen import ModelGenerator
    cfg = pipeline.load_scan_config(None, default_config)
    cfg["prior"] = "active_learning"
    cfg["active_learning"] = {"model": "nope"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(alc.AlConfigError, match="model"):
        ModelGenerator(config_file=str(path), scan_dir=str(tmp_path / "scan"), seed=1, quiet=True)
    cfg["active_learning"] = {"model": "dnn"}
    path.write_text(yaml.safe_dump(cfg))
    mg = ModelGenerator(config_file=str(path), scan_dir=str(tmp_path / "scan"), seed=1, quiet=True)
    assert mg.prior == "active_learning"
    with pytest.raises(ValueError, match="active_learning"):
        mg.generate_models()


def test_shipped_active_learning_configs_validate(datadir):
    for name in ("example_active_learning_config.yaml", "example_active_learning_regression.yaml",
                 "example_active_learning_exclusion.yaml", "ci_active_learning_config.yaml"):
        merged = pipeline.load_scan_config(str(datadir / "configs" / name),
                                           str(datadir / "configs" / "default_config.yaml"))
        cfg = alc.load_al_config(merged)
        assert cfg.environment in alc.ENVIRONMENTS
