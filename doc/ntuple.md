# NTuple structure

Every scan produces a single ROOT file, the **ntuple**, that flattens the output of the
whole pipeline (SPheno, SOFTSUSY, micrOMEGAs, SuperIso, GM2Calc, FeynHiggs, SModelS) into
one table with one row per model. It is the primary analysis product of a scan and the input
to model extraction (`extractModels.py`, see [README](../README.md#extracting-interesting-models)).

The ntuple is written by [`ntupling.py`](../source/Run3ModelGen/python/ntupling.py) using
[uproot](https://uproot.readthedocs.io/). This document describes the file layout, how branch
names are constructed, and what each group of branches contains.

## File layout

```
<scan_dir>/ntuple.<ClusterId>.<ProcId>.root      # or ntuple.0.0.root for local runs
└── susy                                          # a single TTree
    ├── model                                     # branch 0: unique model id
    ├── IN_*                                       # branches from step "prep_input"
    ├── SP_*                                       # branches from step "SPheno"
    ├── SS_*                                       # branches from step "softsusy"
    ├── MO_*                                       # branches from step "micromegas"
    ├── SI_*                                       # branches from step "superiso"
    ├── GM2_*                                      # branches from step "gm2calc"
    ├── FH_*                                       # branches from step "feynhiggs"
    └── SModelS_*                                  # branches from step "SModelS"
```

- **One tree**, always named `susy`.
- **One entry (row) per model.** `num_entries` equals the `num_models` of the scan.
- **All branches are flat `double` arrays** of length `num_models`. There are no nested/jagged
  branches — even branching ratios and mixing-matrix elements are stored as scalar-per-model.
- The set of branches present depends entirely on which `steps` ran (see below). A typical full
  scan has ~1100 branches.

### Missing / error values

Branches are filled lazily: `mkntuple` allocates each branch as
`np.full(num_models, -1.)` the first time a key is seen, then overwrites the entries it can
compute. As a consequence **`-1` is the universal sentinel for "not available"**:

- A model where a tool failed, crashed, or hit a physics error (tachyon, no `MASS` block, etc.)
  keeps `-1` for that tool's physics branches — but its `*_err_*` flags are still set (see
  [Error flags](#error-flags-err_)).
- A NaN read from an SLHA block is also stored as `-1` (with a warning logged).
- When filtering, `x != -1` is the idiomatic "this quantity was successfully computed" cut
  (e.g. the default extraction selection uses `akarr['SS_m_h'] != -1`).

## Branch naming

Every branch (except `model`) is named `<PREFIX>_<variable>`. The prefix comes from the
`prefix` field of the corresponding pipeline step in the scan config (see
[doc/configs.md](configs.md#pipeline-steps)); it is **not** hard-coded, so a config that
renames a step's prefix renames its branches. The default prefixes are:

| Prefix | Step | Tool | Content |
| --- | --- | --- | --- |
| `IN` | `prep_input` | — | pMSSM input parameters (the sampled point) |
| `SP` | `SPheno` | SPheno | Spectrum, mixings, decays, flavour obs, error flags |
| `SS` | `softsusy` | SOFTSUSY | Spectrum, mixings, decays, error flags |
| `MO` | `micromegas` | micrOMEGAs | Relic density, direct detection, low-energy obs |
| `SI` | `superiso` | SuperIso | Flavour observables (BSM and SM) |
| `GM2` | `gm2calc` | GM2Calc | Muon anomalous magnetic moment |
| `FH` | `feynhiggs` | FeynHiggs | Higgs mass (and W mass, if enabled) |
| `SModelS` | `SModelS` | SModelS | LHC re-interpretation limits |

The `model` branch is the exception (no prefix): it is a unique integer id built as
`int(<ProcId> + "00000") + model_index`, so ids stay unique across the `ProcId` shards of a
single `ClusterId`.

## Branch groups

The exact variables read from each tool are defined by the `*_blocks` / `*_decays` dictionaries
at the top of [`ntupling.py`](../source/Run3ModelGen/python/ntupling.py). The mapping is
"SLHA block + key → branch name", so those dictionaries are the authoritative list. The groups
below summarise them.

### Input parameters (`IN_`)

The 24 pMSSM parameters of the sampled point, read back from the generated input SLHA
(`EXTPAR`): `IN_M_1`, `IN_M_2`, `IN_M_3`, `IN_At`, `IN_Ab`, `IN_Atau`, `IN_mu`, `IN_tanb`,
`IN_mA`, and the sfermion soft masses `IN_meL`, `IN_mmuL`, `IN_mtauL`, `IN_meR`, `IN_mmuR`,
`IN_mtauR`, `IN_mqL1`, `IN_mqL2`, `IN_mqL3`, `IN_muR`, `IN_mcR`, `IN_mtR`, `IN_mdR`, `IN_msR`,
`IN_mbR`. Also carries the standard error flags.

### Spectrum & mixings (`SP_`, `SS_`)

SPheno and SOFTSUSY each produce a full spectrum; the two prefixes carry parallel sets of
branches (SPheno additionally has `SPHENOLOWENERGY` flavour observables — see below).

- **Masses (`*_m_*`)** — pole masses from the `MASS` block, e.g. `SP_m_h`, `SP_m_H`, `SP_m_A`,
  `SP_m_Hp`, `SP_m_W`, the sfermions (`SP_m_t_1`, `SP_m_b_1`, `SP_m_tau_1`, `SP_m_e_L`, …), the
  gluino `SP_m_gl`, neutralinos/charginos (`SP_m_chi_10`…`SP_m_chi_40`, `SP_m_chi_1p`,
  `SP_m_chi_2p`) and the gravitino `SP_m_G`.
- **Mixing matrices** — neutralino `N_ij` (`SP_N_11`…`SP_N_44`), chargino `U_ij`/`V_ij`, and the
  third-generation sfermion mixings `SP_cos_t`, `SP_cos_b`, `SP_cos_tau`.
- **Also present:** `tanb_min` (`MINPAR`) and `tanb_ext` (`EXTPAR`).

#### LSP / LSQ and gaugino fractions (mass-block steps only)

Computed only for steps flagged `needsMass` (SPheno, SOFTSUSY):

- `*_LSP_type`, `*_LSP_mass` — the lightest SUSY particle. `LSP_type` is a PDG id, but when the
  LSP is the lightest neutralino (`1000022`) it is **re-coded** to its dominant composition:
  `1` = Bino-like, `2` = Wino-like, `3` = Higgsino-like.
- `*_LSQ_type`, `*_LSQ_mass` — the lightest squark.
- `*_LSP_Bino_frac` / `*_LSP_Wino_frac` / `*_LSP_Higgsino_frac` — LSP composition (only when the
  LSP is a neutralino).
- Per-state gaugino fractions for all neutralinos and charginos, derived from the mixing
  matrices: `*_chi_10_Bino_frac`, `*_chi_10_Wino_frac`, `*_chi_10_Higgsino_frac`, … through
  `chi_40`, plus `*_chi_1p_Wino_frac` / `*_chi_1p_Higgsino_frac` (and `chi_2p`, `chi_1m`,
  `chi_2m`). These are always computed (not just for the LSP).

#### Decays — widths and branching fractions (`*_w_*`, `*_BF_*`)

For each parent particle in the `softsusy_decays` table, the ntuple stores:

- `*_w_<parent>` — the total width (`-1` if the particle has no `DECAY` block).
- `*_BF_<parent>_to_<finalstate>` — the branching fraction into each **explicitly listed**
  final state, with particle names from the `PDG` map (e.g. `SP_BF_gl_to_d_L`,
  `SP_BF_chi_20_to_chi_10_Z`, `SP_BF_h_to_b_b`). Matching is by PDG-id **subset**, so a listed
  2-body channel absorbs the corresponding 3-body cascades.
- `*_BF_<parent>_other` — the summed branching fraction into everything not explicitly listed.

This is the bulk of the branch count (~300 `BF_` branches per spectrum tool).

### micrOMEGAs (`MO_`)

Read from a CSV (`Variable,MO` columns). Twelve observables:
`MO_Omega` (relic density Ωh²), `MO_Xf` (freeze-out `x_f`), the direct-detection cross
sections `MO_proton_SI`, `MO_proton_SD`, `MO_neutron_SI`, `MO_neutron_SD`, and low-energy
observables `MO_deltarho`, `MO_gmuon`, `MO_bsgnlo`, `MO_bsmumu`, `MO_btaunu`, `MO_Rl23`.

### SuperIso (`SI_`)

Flavour observables from the `FOBS` (BSM) and `FOBSSM` (SM reference) blocks, e.g.
`SI_BR_b_to_sgamma`, `SI_BR_Bs_to_mumu`, `SI_Delta0_B_to_Kshgamma`, `SI_R_Bu_to_taunu`,
`SI_R_mu23`, and their `_SM` counterparts. Also carries error flags.

### GM2Calc (`GM2_`)

`GM2_gmuon` (the MSSM contribution to (g−2)_μ / 2) and its uncertainty `GM2_Delta_gmuon`,
from the `GM2CALCOUTPUT` block. Plus error flags.

### FeynHiggs (`FH_`)

`FH_m_h` (Higgs mass) and `FH_Delta_m_h` (its uncertainty). If the W-mass variant is enabled,
`FH_m_w_mssm` and `FH_m_w_sm` from the `PRECOBS` block. Plus error flags.

### SModelS (`SModelS_`)

Read from SModelS's Python output. For each analysis that produced a result:

- `SModelS_<AnalysisID>_TheoryPrediction` — predicted signal cross section (fb).
- `SModelS_<AnalysisID>_UpperLimit` — observed upper limit (fb).
- `SModelS_<AnalysisID>_ExpUpperLimit` — expected upper limit (fb).

`AnalysisID` has dashes replaced by underscores (e.g. `CMS_SUS_19_006`). In addition, the
single most sensitive analysis (lowest expected UL) is duplicated under the
`SModelS_bestExpUL_*` branches. A model is excluded by an analysis when its
`TheoryPrediction > UpperLimit`.

### Error flags (`*_err_*`)

Every SLHA-based step (`IN`, `SP`, `SS`, `SI`, `GM2`, `FH`) carries the same block of integer
flags, set to `1` when the corresponding problem was detected in the tool's raw output and `0`
otherwise:

`err_tachyon` (with sub-flags `err_tachyon_sbottom`, `_stop`, `_snutau`, `_A0`, `_stau`,
`_hpm`, `_m3sq_problem`, `_other`), `err_m3sq_problem`, `err_stop_LSP`, `err_higgs_potential`,
`err_num`, `err_other`. There is also a single global `Maywarning` flag.

When any error is flagged the model's physics branches for that step stay at the `-1` sentinel,
so `err_*` and the `-1` convention together let you distinguish "tool ran and produced a real
value", "tool ran but flagged a physics problem", and "tool did not run".

## Reading the ntuple

With uproot / awkward (as `extractModels.py` does):

```python
import uproot
tree = uproot.open("scan/ntuple.0.0.root")["susy"]

# read selected branches as an awkward record array
akarr = tree.arrays(["model", "SP_m_h", "MO_Omega", "SP_LSP_type"])

# viable-Higgs, neutralino-LSP, relic-density-not-overclosing example cut
mask = (akarr["SP_m_h"] != -1) & (akarr["SP_m_h"] <= 130) & (akarr["MO_Omega"] < 0.12)
selected = akarr[mask]
```

Any ROOT-aware tool works too (`ROOT.RDataFrame`, `TTree::Draw`, `pandas` via
`uproot.arrays(library="pd")`).

## Where the structure is defined

- Branch definitions (block → name maps, decay tables, PDG names): the module-level dicts in
  [`ntupling.py`](../source/Run3ModelGen/python/ntupling.py)
  (`softsusy_blocks`, `spheno_blocks`, `superiso_blocks`, `gm2calc_blocks`, `feynhiggs_blocks`,
  `input_blocks`, `softsusy_decays`, `PDG`).
- Row assembly per model: `readModel()`; per-file SLHA parsing and error detection:
  `readSLHA()`.
- Tree assembly and file writing: `mkntuple()`.
- Which steps (and therefore which prefixes/branches) run: the `steps` list in the scan config,
  documented in [doc/configs.md](configs.md#pipeline-steps).
