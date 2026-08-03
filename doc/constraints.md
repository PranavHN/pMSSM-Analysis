# Constraints

Constraints define how physics observables are compared to experimental measurements. They are used by [MCMC sampling](emcee.md) to compute a chi-squared (chi2) value for each model, which drives the emcee log-probability.

## Configuration

Constraints are specified as a list in the config YAML file:

```yaml
constraints:
  - name: higgs_mass
    source:
      step: SPheno
      file_type: slha
      block: MASS
      entry: 25
    type: gaussian
    central: 125.09
    uncertainty: 3.0

  - name: relic_density
    source:
      step: micromegas
      file_type: csv
      variable: Omega
    type: upper_bound
    central: 0.120
    uncertainty: 0.012
```

## Constraint fields

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | A descriptive name for the constraint (used in logging). |
| `source` | Yes | Specifies where to read the observable from. See [Source specification](#source-specification). |
| `type` | Yes | The type of chi2 contribution. See [Constraint types](#constraint-types). |
| `central` | Yes | The central/reference value of the measurement. |
| `uncertainty` | Yes | The 1-sigma uncertainty of the measurement. |

## Source specification

The `source` block tells the MCMC where to find the observable value in the pipeline output.

| Field | Required | Description |
| --- | --- | --- |
| `step` | Yes | The pipeline step name (e.g. `SPheno`, `softsusy`, `micromegas`, `superiso`). |
| `file_type` | Yes | The output file format: `slha`, `flha`, or `csv`. |
| `block` | For slha/flha | The SLHA/FLHA block name (e.g. `MASS`, `FOBS`). |
| `entry` | For slha/flha | The entry key. An integer for single keys (e.g. `25`), or a list for tuple keys (e.g. `[5, 1]`). |
| `variable` | For csv | The variable name in the CSV file (e.g. `Omega`). |

### File type examples

**SLHA** (SPheno, softsusy, feynhiggs, gm2calc):
```yaml
source:
  step: SPheno
  file_type: slha
  block: MASS
  entry: 25        # Higgs mass (PDG ID 25)
```

**FLHA** (superiso):
```yaml
source:
  step: superiso
  file_type: flha
  block: FOBS
  entry: [5, 1]   # BR(b -> s gamma), tuple key
```

**CSV** (micromegas):
```yaml
source:
  step: micromegas
  file_type: csv
  variable: Omega  # Relic density
```

## Constraint types

### `gaussian`

Penalises deviations in both directions from the central value:

```
chi2 = ((observed - central) / uncertainty)^2
```

Use for measurements with a known value, e.g. the Higgs mass.

### `upper_bound`

Only penalises if the observed value exceeds the central value:

```
chi2 = ((observed - central) / uncertainty)^2   if observed > central
chi2 = 0                                        if observed <= central
```

Use for experimental upper limits, e.g. the dark matter relic density.

### `lower_bound`

Only penalises if the observed value falls below the central value:

```
chi2 = ((central - observed) / uncertainty)^2   if observed < central
chi2 = 0                                        if observed >= central
```

Use for experimental lower limits.

## Total chi2

The total chi2 for a model is the sum of all individual constraint contributions:

```
chi2_total = sum(chi2_i for each constraint i)
```

The likelihood is then `L = exp(-chi2_total / 2)`, and the emcee log-probability sampled by `prior: mcmc` is `-chi2_total / 2 + flat_prior` (out-of-bounds or pipeline-failed points return `-inf`).

## Common constraint examples

```yaml
constraints:
  # Higgs boson mass (combined ATLAS+CMS measurement)
  - name: higgs_mass
    source:
      step: SPheno
      file_type: slha
      block: MASS
      entry: 25
    type: gaussian
    central: 125.09
    uncertainty: 3.0

  # Dark matter relic density (Planck upper bound)
  - name: relic_density
    source:
      step: micromegas
      file_type: csv
      variable: Omega
    type: upper_bound
    central: 0.120
    uncertainty: 0.012

  # BR(b -> s gamma) (world average)
  - name: bsg_branching
    source:
      step: superiso
      file_type: flha
      block: FOBS
      entry: [5, 1]
    type: gaussian
    central: 3.32e-4
    uncertainty: 0.33e-4

  # BR(Bs -> mu mu) (LHCb + CMS combination)
  - name: bs_mumu
    source:
      step: superiso
      file_type: flha
      block: FOBS
      entry: [531, 1]
    type: gaussian
    central: 3.09e-9
    uncertainty: 0.65e-9
```
