'''prior: active_learning — surrogate-guided sampling inside Run3ModelGen.

The loop (see :mod:`Run3ModelGen.active_learning.loop`) labels an initial uniform
sample with the physics pipeline, trains a surrogate on the labelled set, scores a
fresh pool of candidate points, selects a batch by an acquisition rule, labels the
batch, and repeats. Every iteration is an ordinary scan directory
(``<scan_dir>/iteration_NNN/``) with the per-tool output dirs, ``scan_config.yaml``
and ``ntuple.0.0.root``; the scan's own ntuple is the merge of all iterations plus
``al_*`` branches. It is driven by the ``active_learning`` Snakemake rule
(``workflow/rules/active_learning.smk`` -> ``r3mg_batch.py active_learning``).

Submodules that never import torch, usable in the default (CPU) environment and
by the test suite: :mod:`.config`, :mod:`.data`, :mod:`.labelling`, :mod:`.ntuple`.
Everything else needs the ``rocm`` or ``cuda`` pixi environment.

The surrogates, heads and acquisition rules are ports of the pmssm-al-benchmark
repository (https://github.com/jwuerzinger/pmssm-al-benchmark, MIT); the GP models
and the entropy-batch strategy are vendored from al_pmssmwithgp (see gp.py).
Nothing here imports either repository.
'''
