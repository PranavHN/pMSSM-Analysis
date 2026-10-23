# Active learning — `prior: active_learning`

`prior: active_learning` makes Run3ModelGen sample where a surrogate model says the
interesting models are. The loop labels a uniform seed set with the usual pipeline,
trains a surrogate on every labelled model, scores a fresh pool of candidate points,
selects a batch by an acquisition rule, labels the batch, and repeats:

```
iteration_000     seed_models uniform draws from `parameters`, run through the pipeline
iteration_001..N  train surrogate on all valid labelled models (80/20 train/validation)
                  draw n_candidates (Latin hypercube or uniform) in the free parameter box
                  score them; select num_models by the acquisition rule; run the pipeline on them
                  retry with a fresh pool until min_valid_fraction x num_models came out valid
                  append the batch to the labelled set; checkpoint al_state.pt
```

It is a port of the loop benchmarked in the pMSSM active-learning paper
([jwuerzinger/pmssm-al-benchmark](https://github.com/jwuerzinger/pmssm-al-benchmark)),
with the same surrogates, heads, acquisition rules and hyper-parameters, minus the paper's
evaluation machinery (random-baseline model, static evaluation sets, MCMC reference).
Run3ModelGen does not depend on that repository: everything lives in
`source/Run3ModelGen/python/active_learning/` (the GP models and the entropy-batch
selector are vendored from
[al_pmssmwithgp](https://github.com/jwuerzinger/al_pmssmwithgp), see `gp.py`).

## The GPU environments, and why the default one refuses

The surrogate trains on a GPU; **there is no CPU mode.** `pixi.toml` defines two extra
environments, `rocm` (AMD; MPCDF viper's MI300A nodes) and `cuda` (NVIDIA), both
*supersets* of the default environment: the physics tools plus `torch`/`gpytorch`. The
loop labels its batches with the pipeline inside the same job, so the tools must be in
the same environment.

```bash
pixi install -e rocm            # or -e cuda; ~25 GB (torch + ROCm libraries)
pixi run -e rocm build          # the build is environment-independent: once is enough
pixi run -e rocm genModels.py --config_file source/Run3ModelGen/data/configs/example_active_learning_config.yaml \
    --scan_dir /ptmp/$USER/al_relic --backend slurm
```

Which environment a scan uses is a key of the scan config
(`active_learning.environment: rocm | cuda`); the Snakemake rule shells of that run
execute in it. The default (CPU) environment is refused at three points, each with a
line starting `FATAL`:

* `environment: default` fails config validation;
* `genModels.py` launched from any other environment than the configured one
  (`pixi run genModels.py ...` from the default environment, say) fails its pre-flight
  before anything is wiped or submitted, as does a configured environment that is not
  installed or cannot import torch;
* the rule body itself exits 1 if torch sees no GPU (a `--backend local` run on a
  login node, for instance).

The `cuda` environment is defined and lock-solved but has not been exercised on an
NVIDIA machine yet.

## Configuration

Every knob lives in the `active_learning` block of the scan config, with defaults in
[`default_config.yaml`](../source/Run3ModelGen/data/configs/default_config.yaml);
a user config overrides any subset of the keys (nested keys merge, lists replace).
The top-level `num_models` is the number of points selected and labelled per iteration.
`tool_timeout` applies to every external tool call, as for `prior: mcmc`.

| key | default | meaning |
| --- | --- | --- |
| `environment` | `rocm` | pixi environment the loop runs in: `rocm` or `cuda` (never `default`) |
| `nworkers` | 16 | CPU workers labelling one batch; = the rule's cpus-per-task |
| `walltime_hours` | 23 | wall-clock guard: the loop stops *between* iterations when the next one would not fit, and resumes on the next run |
| `seed_models` | 2000 | size of the uniform seed set (iteration_000) |
| `n_iterations` | 40 | number of active-learning iterations after the seed set |
| `val_fraction` | 0.2 | fraction of every labelled batch that goes to the validation split |
| `model` | `transformer` | `transformer`, `dnn`, `exact_gp`, `deep_gp` |
| `head` | `classification` | `classification` (learn which side of `target.value` a point falls on) or `regression` (learn the transformed target) |
| `strategy` | `cls_entropy` | classification: `cls_entropy`; regression: `entropy_batch` or `tol_only_random` |
| `target.branch` | `MO_Omega` | ntuple branch the surrogate learns |
| `target.value` | 0.12 | the target value: `t = log(branch / value)`, classification label `branch > value` |
| `target.transform` | `log` | `log` or `none` (then `t = branch` and the threshold is `value`) |
| `target.valid` | `{lo: 0, hi: 1}` | strict window the raw branch must satisfy to count as valid; the `-1` "not computed" sentinel never passes |
| `target.hit_tolerance` | 0.10 | `|branch - value| / |value| < hit_tolerance` is a *hit* (diagnostics only) |
| `vetoes` | `SP_m_h > 0`, `0.5 < SP_LSP_type < 3.5` | strict windows on further branches; every one must pass for a model to be valid (the `hard_window` semantics of the MCMC constraints) |
| `candidates.n_candidates` | 1 000 000 | candidates scored per iteration |
| `candidates.generation` | `lhs` | `lhs` (Latin hypercube) or `uniform`, over the free parameters |
| `training.epochs`, `.patience` | 10000, 200 | epochs (GP: iterations) and early-stopping patience on the validation loss |
| `training.warm_start` | false | start from the previous iteration's weights (the paper runs the GPs warm, the neural surrogates cold) |
| `training.learning_rate`, `.weight_decay`, `.batch_size` | 3e-4, 1e-4, 256 | neural training (AdamW, cosine annealing to 1e-6, gradient clip 1.0) |
| `transformer` | d_model 128, nhead 4, num_layers 3, dim_feedforward 512, dropout 0.1 | the MC-dropout transformer |
| `dnn` | d_model 64, num_layers 4, dim_feedforward 256, dropout 0.1 | the MC-dropout feed-forward network |
| `gp` | RBF, lengthscale 1, noise 0.01, jitter 1e-3, ARD, lr 1e-3, batch 256, 8 samples, 10 hidden dims, 256 inducing | kernel and training of the exact and deep GP |
| `acquisition.mc_samples` | 30 | MC-dropout passes per candidate (neural surrogates) |
| `acquisition.tolerance` | 1.0 | regression rules keep candidates with `|mean - threshold| < tolerance` first (0 = off) |
| `acquisition.proximity` | 0.1 | `entropy_batch` weights the variance by `exp(-(mean - threshold)^2 / proximity)` (0 = off) |
| `acquisition.entropy_pool_size`, `.entropy_blur`, `.entropy_beta` | 5000, 0.15, 50 | `entropy_batch`: focused pool size, boundary blur, Gibbs temperature |
| `labelling.min_valid_fraction`, `.max_attempts` | 0.6, 10 | an iteration retries with a fresh candidate pool until this fraction of `num_models` came out valid, at most this often |

Validation names the offending key (`active_learning.<key>: ...`) and runs at config load
in any environment. Supported combinations:

| model | head | strategy |
| --- | --- | --- |
| transformer, dnn | classification | cls_entropy |
| transformer, dnn | regression | entropy_batch, tol_only_random |
| deep_gp | classification (Bernoulli likelihood, probit) | cls_entropy |
| exact_gp | classification (least-squares +-1 verdict, probit read-out) | cls_entropy |
| exact_gp, deep_gp | regression | entropy_batch, tol_only_random |

Shipped examples: `example_active_learning_config.yaml` (relic density, transformer
classification, the default), `example_active_learning_regression.yaml` (deep GP
regression, entropy batch, warm start), `example_active_learning_exclusion.yaml`
(SModelS best expected r-value, DNN regression, tolerance cut + random) and the tiny
`ci_active_learning_config.yaml` used by the DAG dry-run and the GPU smoke test.
`genConfig.py active_learning` copies the first one.

### The heads and the rules

* **classification** trains a logit of `1[t > 0]` (binary cross-entropy for the neural
  surrogates; a Bernoulli-likelihood deep GP; for the exact GP the +-1 verdict regressed
  under the Gaussian likelihood, read through a probit link, the only way to keep exact
  inference). `cls_entropy` ranks every candidate by the predictive entropy
  `H[mean p]` of the MC-dropout (or posterior) committee and takes the top `num_models`:
  the points whose verdict is least certain.
* **regression** trains on `t` with a squared loss; the uncertainty is the MC-dropout
  variance or the GP posterior variance. `tol_only_random` applies the tolerance cut and
  draws `num_models` survivors uniformly (the control arm that measures what the cut
  alone contributes). `entropy_batch` applies the cut, weights the variance by the
  proximity to the threshold, keeps the `entropy_pool_size` best, and selects a diverse
  batch by the smoothed batch entropy of the posterior over that pool (sample covariance
  of the MC passes for the neural surrogates, the predictive covariance for the GPs),
  Gibbs-sampling each next point at temperature `entropy_beta`.

Inputs are the free parameters only (`lo != hi` in `parameters`), z-scored with the
training split for the neural surrogates and min-max scaled to the free box for the GPs.
Only *valid* models (target window and every veto) enter the training set.

## Running

```bash
# single scan dir, one GPU job on SLURM (the slurm profile puts the active_learning rule
# on the apu1 partition with --gres=gpu:1; adapt workflow/profiles/slurm/config.yaml)
pixi run -e rocm genModels.py --config_file al.yaml --scan_dir /ptmp/$USER/al_run --backend slurm

# K independent scans with their own seeds (scan.<runid>.<b>), e.g. for seed averages
pixi run -e rocm genModels.py --config_file al.yaml --scan_dir /ptmp/$USER/al_runs --backend slurm --num_jobs 5

# on a GPU machine, in the foreground
pixi run -e rocm genModels.py --config_file al.yaml --scan_dir al_run
```

Resources: the rule requests `nworkers` CPUs and `nworkers * 2000 + 8000` MB
(SModelS in the steps needs more: every worker loads the 1.4 GB results database;
`--set-resources active_learning:mem_mb=<n>`). A full run of 40 iterations with 500
points each is hours to days; the loop **checkpoints after every iteration and
resumes** when the same command is run again (the state `al_state.pt` and the
`iteration_NNN/` dirs are side-state the workflow never deletes, like `chain.h5`).
A wall-clock stop exits the rule with code 3 *without* the scan's ntuple, so Snakemake's
retries or your re-run continue it; the fresh-scan wipe of `genModels.py --num_jobs 1`
refuses a scan dir that holds `al_state.pt` (`R3MG_FORCE_WIPE=1` to discard it).
`rerunStep.py` refuses active-learning scans: every batch is labelled inside the loop.

## Output

```
<scan_dir>/
  ntuple.<C>.<P>.root      every labelled model of every iteration (see below)
  scan_config.yaml         the merged config, scan dir and seed (provenance, as always)
  active_learning.log      the loop's log (also in the job's log)
  al_state.pt              checkpoint: the labelled set, roles, RNG state, warm-start weights
  al_summary.json          per-iteration counts, timings, training metrics
  diag/                    hit_rate.png, training.png, diagnostics.txt, summary.json
  iteration_000/           the seed set — an ordinary scan dir:
    input/ SPheno/ SPheno_log/ micromegas/ ...   per-tool outputs, model index = file name
    ntuple.0.0.root        this iteration's models (ids 0..n-1)
    scan_config.yaml       provenance
    steps.json             per-step counts, per labelling attempt (the manifest schema)
    selected_points.csv    one row per labelled point: index, attempt, all 19 parameters, pred, score
    selection.json         strategy, candidate-pool summary and per-model prediction/score/valid/hit
    al_iteration.json      counts, hits and timings of the iteration
  iteration_001/           as above, plus
    training.log           the epoch-by-epoch training log
    model.pt               the surrogate checkpoint (state dicts)
    metrics.json           losses per epoch, best epoch, verdict accuracy (R2/MSE for regression)
  ...
```

Every `iteration_NNN/` is a complete scan dir, so the usual tools apply to it:
`extractModels.py --scan_dir <scan>/iteration_007 --root_file <scan>/iteration_007/ntuple.0.0.root ...`.
The labelling attempts of one iteration share the dir, with model indices continuing
(`attempt 0` labels `0 .. num_models-1`, attempt 1 the next block, ...), so one ntuple
holds every attempt; `selection.json` and `steps.json` record which attempt a model
came from.

The scan's ntuple is the merge of the iteration ntuples: the union of their branches,
`-1` where a branch is absent, `model = b * 100000 + running index` (the batch
contract), plus

| branch | meaning |
| --- | --- |
| `al_iteration` | iteration that labelled the model (0 = seed set) |
| `al_attempt` | labelling attempt inside the iteration (0 = the selected batch) |
| `al_index` | the model's index inside `iteration_NNN/` |
| `al_valid` | 1 if the target window and every veto passed |
| `al_hit` | 1 if inside the hit band |
| `al_pred` | the surrogate's prediction in transformed space when the point was selected (-1 for seeds) |
| `al_score` | the acquisition score when the point was selected (-1 for seeds and for `tol_only_random`) |

## Reproducibility and seeds

A scan is seeded by `--seed` (batch mode: the per-batch `blake2b` seed, as for every
prior): the seed set, the candidate pools (`seed * 10000 + iteration`, retries
`+ iteration * 1000 + attempt`), the `tol_only_random` draws (`seed * 100000 +
iteration`), the train/validation split, the model initialisation and the Gibbs draws
of `entropy_batch` all derive from it, and the RNG states are checkpointed. GPU
training is not bit-reproducible across runs (non-deterministic kernels), so two runs at
the same seed agree in distribution, not bit for bit; `prior: active_learning` is
therefore not value-gated in CI, and CI has no GPU to run it at all. Its coverage is the
unit suite (`tests/test_active_learning_*.py`; the torch half runs on the CPU in the
GPU environments: `pixi run -e rocm test`) and the DAG dry-run in `tests/run_checks.sh`.

## Differences from the benchmark loop

Deliberate, all small: the seed set is a fresh uniform sample rather than a slice of a
pre-existing random-scan pool; the surrogates see the free parameters only (the
benchmark fed all 19, the 10 fixed ones z-scored to a constant); the GPs normalise with
the config's parameter box instead of a fixed range table; the run seed reaches the GP
constructors (the benchmark re-seeded with 42); a point drawn twice by the entropy
batch is labelled once; and each iteration's labelling attempts live in one scan dir
instead of `retry_NNN/` sub-dirs. The random-baseline model, the static and MCMC
evaluation sets, the oracle candidate source, ensembles, Laplace uncertainty and the
`top_k`/`bald` strategies are not ported.
