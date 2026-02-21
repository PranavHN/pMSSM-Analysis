# MCMC Sampling

This document describes the Markov Chain Monte Carlo (MCMC) sampling feature, which allows physics-informed exploration of the pMSSM parameter space using the Metropolis-Hastings algorithm.

## Overview

Unlike flat sampling (which draws parameters uniformly at random), MCMC sampling uses a likelihood function derived from physics observables to preferentially explore regions of parameter space that agree with experimental measurements. The likelihood is defined via [constraints](constraints.md) on observables computed by the physics pipeline.

## Algorithm

The implementation uses the **Metropolis-Hastings** algorithm:

1. **Initialisation**: Draw a random starting point uniformly within the parameter bounds.

2. **Proposal**: Perturb the current point with a Gaussian: `x' = x + N(0, sigma)`, where `sigma` is the per-parameter proposal width. If the proposed value falls outside bounds, it is reflected back in to preserve detailed balance.

3. **Pipeline evaluation**: Run the full physics pipeline (prep_input, SPheno, softsusy, ...) on the proposed point. If any step fails, assign a penalty chi2 of 10^6.

4. **Acceptance criterion**:
   - Compute `delta_chi2 = chi2(proposed) - chi2(current)`
   - If `delta_chi2 <= 0` (proposed is better or equal): always accept
   - If `delta_chi2 > 0` (proposed is worse): accept with probability `exp(-delta_chi2 / 2)`

5. **Collection**: If accepted and past the burn-in phase, save the model to the output. Otherwise, the chain moves but the model is not saved.

6. Repeat from step 2 until `num_models` accepted models are collected or `max_attempts` is reached.

## Configuration

To use MCMC sampling, set `prior: mcmc` in your config file and add an `mcmc` block and a `constraints` list. A full example can be generated with:

```bash
genConfig.py mcmc
```

### MCMC block

```yaml
mcmc:
  burn_in: 100          # Number of accepted steps to discard before collecting
  max_attempts: 50000   # Hard cap on total proposals (includes burn-in)
  accept_invalid: false # Whether to save models where the pipeline failed
  proposal_widths:      # Optional per-parameter Gaussian widths
    tanb: 3.0
    M_1: 100.0
```

| Key | Default | Description |
| --- | --- | --- |
| `burn_in` | `0` | Number of accepted steps to discard at the start. Allows the chain to settle into a high-likelihood region before collecting models. |
| `max_attempts` | `10000` | Maximum total proposals (including burn-in). The run stops when either `num_models` are collected or this limit is reached. |
| `accept_invalid` | `false` | If `false`, models where the pipeline failed (chi2 = 10^6) are used for chain movement but not saved. If `true`, they are saved like any other accepted model. |
| `proposal_widths` | `{}` | Per-parameter Gaussian standard deviations. Parameters not listed default to 5% of their range. |

### Proposal widths

The proposal width controls how far the chain jumps in each parameter per step. This directly affects the acceptance rate:

- **Too large**: proposals land in unlikely regions, most are rejected, the chain moves slowly.
- **Too small**: almost everything is accepted, but the chain explores slowly via tiny steps.
- **Rule of thumb**: aim for a 25-50% acceptance rate.

If not specified, each parameter's width defaults to **5% of its total range**. For example, `M_1: [-2000, 2000]` (range = 4000) gets a default width of 200.

### Constraints

See [constraints.md](constraints.md) for the full constraint specification.

## Output

MCMC output is identical in format to flat sampling:
- Input SLHA files at `scan/input/{0..num_models-1}.slha`
- Pipeline output files at `scan/{step_output_dir}/{0..num_models-1}.{ext}`
- ROOT ntuple at `scan/ntuple.0.0.root`
- Scan config dump at `scan/scan_config.yaml`

Only accepted (and valid, if `accept_invalid: false`) models appear in the output. The scratch workspace used during sampling is cleaned up automatically.

## Output validation

To ensure consistency between the MCMC acceptance and the ROOT ntuple, `run_SPheno` and `run_softsusy` validate their output files using the same `pyslha` parsing that the ntupling code uses. If `pyslha` cannot parse a step's output (e.g. due to a charged LSP warning in the SPINFO block), the step is marked as failed and the model receives a penalty chi2.

## Logging

The MCMC produces structured log output:

```
MCMC burn-in starting (100 accepted steps)...
MCMC burn-in complete after 100 accepted steps
MCMC accepted 1/500 (attempt 142, file=scan/input/0.slha, chi2=0.42)
MCMC accepted 2/500 (attempt 143, file=scan/input/1.slha, chi2=0.38)
...
MCMC progress: 98/500 accepted, 200 total attempts, acceptance rate: 49.0%
...
MCMC complete: 500 models, 1023 proposals, rate=48.9%
```

If `max_attempts` is reached before collecting `num_models`, a warning is logged and `num_models` is adjusted to the actual count.
