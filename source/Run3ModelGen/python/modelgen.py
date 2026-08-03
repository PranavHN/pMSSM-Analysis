'''Module for ModelGenerator class'''

import math
import yaml
import numpy as np
import os
datadir = os.environ['DATAPATH'].split(':')[0]

import pyslha

from Run3ModelGen import ntupling
from Run3ModelGen.ntupling import mkntuple
from Run3ModelGen import pipeline
from Run3ModelGen import sampling
from Run3ModelGen import steps as toolsteps
from Run3ModelGen.steps import StructlogHandler  # re-exported for backwards compatibility
import Run3ModelGen.csvReader as csvReader

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)  # so we have logger names

class ModelGenerator:
    '''Class for Model Generation.'''
    def __init__(self, config_file: str = None, scan_dir: str = f"scan", seed: int = 123, quiet: bool = False) -> None:
        '''Initialise scan. Set quiet=True to suppress the logo/verbose init logging
        (used by the per-step workflow rules, which construct one ModelGenerator per
        rule — only the batch-opening sample/chain rules print the banner).'''

        self.quiet = quiet

        # Print logo. Note: ASCII art generated with https://patorjk.com/software/taag/ (Small, Fitted)
        if not quiet:
            with open(f"{datadir}/logo.txt", 'r') as file:
                logo = "\n"+file.read()+"\n"
            print(logo)

        self.config_file = config_file
        self.rawfilen = f"{datadir}/raw.slha"
        self.points = {}

        # Add all keys in config_file as attributes. Replace attribute by default if not
        # provided; extra keys not in the default (e.g. 'grid') are carried through.
        default_config = f"{datadir}/configs/default_config.yaml"
        merged = pipeline.load_scan_config(self.config_file, default_config)
        for key, value in merged.items():
            setattr(self, key, value)

        # A config carrying these keys is almost always a _dump_scan_config() output being
        # reused as an input. Harmless now that the caller wins below, but say so: the
        # values in the file are NOT what this run will use.
        reused = [k for k in ('scan_dir', 'seed') if k in merged]
        if reused and not quiet:
            log.warning(
                f"config {self.config_file} carries provenance key(s) {reused} — ignoring "
                f"them in favour of the caller's scan_dir={scan_dir!r}, seed={seed!r}. "
                f"(Looks like a dumped scan_config.yaml reused as an input config.)")

        # CALLER-OWNED, set AFTER the config loop so the config can never override them.
        # _dump_scan_config() writes scan_dir/seed into <scan_dir>/scan_config.yaml as
        # PROVENANCE (where this scan ran, with which seed). Feeding such a dump back in
        # as --config_file used to silently redirect every file operation -- including the
        # fresh-scan wipe in r3mg_batch.py sample -- at the recorded scan dir rather than
        # the one the caller asked for. They are outputs of a scan, never inputs to one.
        self.scan_dir = scan_dir
        self.seed = seed

        # Drop pipeline steps whose external tool is not installed on this platform
        # (e.g. FeynHiggs is not packaged on conda-forge for macOS).
        self.steps = pipeline.resolve_steps(getattr(self, 'steps', []))

        # Check if too many models supplied
        pipeline.check_model_cap(self.num_models)
        
        # Set numpy seed:
        np.random.seed(self.seed)

        # Derive the list of free (varying) parameters; fixed params have lo == hi.
        # Used by the emcee sampler (prior: mcmc) to define the
        # sampled dimensions.
        self.free_names = [k for k, v in self.parameters.items() if not self._is_fixed(v)]

        # Optional wall-clock timeout (seconds) for external physics-tool calls, read
        # from the config if present (threaded into steps.py via run_step). Guards
        # against a single hung binary stalling a whole emcee half-step. Default None
        # keeps the point-sampled priors' behaviour unchanged.
        self.tool_timeout = getattr(self, 'tool_timeout', None)

        # Fail fast on a config whose constraint / record_observables sources reference a
        # step that does not run (else _extract_observable returns None for every point and
        # the emcee chain never moves), or a non-positive constraint uncertainty.
        self._validate_observable_sources()

        # Print information on initialised generator
        if not quiet:
            log.info(f"Initialised ModelGenerator with:")
            for var, varval in vars(self).items():
                if var == 'parameters':
                    log.info(f"\t{var} = ")
                    for param, paramval in varval.items(): log.info(f"\t\t{param}: {paramval}")
                elif var == 'steps':
                    log.info(f"\t{var} = ")
                    for step in varval: log.info(f"\t\t{step}")
                else:
                    log.info(f"\t{var} = {varval}")

        return None
    
    def read_yaml_file(self, file_path: str) -> dict:
        '''Function for reading yaml file. Returns dict.'''
        return pipeline.read_yaml_file(file_path)

    def _setup_scan_dirs(self) -> None:
        '''Create the per-step output directory tree under self.scan_dir.
        Idempotent (exist_ok=True) — the destructive wipe of a pre-existing
        scan dir stays in generate_models(), NOT here, so callers that own
        incremental outputs (the Snakemake rules) can reuse this safely.'''
        os.makedirs(self.scan_dir, exist_ok=True)
        for step in self.steps:
            for dirn in ('output_dir', 'log_dir'):
                if dirn not in step:
                    continue
                os.makedirs(f"{self.scan_dir}/{step[dirn]}", exist_ok=True)
                if "micromegas" in step['name']:
                    os.makedirs(f"{self.scan_dir}/{step[dirn]}_raw", exist_ok=True)
                if "evade" in step['name']:
                    os.makedirs(f"{self.scan_dir}/{step[dirn]}_config", exist_ok=True)
                    os.makedirs(f"{self.scan_dir}/{step[dirn]}_evade_in", exist_ok=True)

    def _init_points_arrays(self, size: int = 1) -> None:
        '''Allocate self.points arrays (one per parameter) of the given size.'''
        for key in self.parameters:
            self.points[key] = np.zeros(size)

    # ── emcee support: fixed-point fill + process-safe single-point evaluation ──
    @staticmethod
    def _is_fixed(scanrange) -> bool:
        '''A parameter is fixed if it has a scalar [lo, hi] range with lo == hi.
        Interrupted ranges (list-of-[lo,hi]) are never fixed.'''
        if isinstance(scanrange[0], list):
            return False
        return scanrange[0] == scanrange[1]

    def _init_fixed_points(self, modelnum: int = 0) -> None:
        '''Write fixed-parameter values (lo == hi) into self.points[...][modelnum].
        Free / interrupted-range params are overwritten per-evaluation by
        evaluate_theta.'''
        for key, scanrange in self.parameters.items():
            if self._is_fixed(scanrange):
                self.points[key][modelnum] = scanrange[0]

    def evaluate_theta(self, theta, record_sources: list = None, modelnum: int = 0):
        '''Run the full pipeline for a single free-parameter vector `theta`
        (ordered as self.free_names) and return (chi2, observables_dict).

        Reuses _run_and_evaluate (steps + chi2 from constraints) and
        _extract_observable (per-observable readback). This is the single reuse
        hook the emcee sampler calls; it never touches SLHA directly.'''
        for name, val in zip(self.free_names, theta):
            self.points[name][modelnum] = val
        chi2 = self._run_and_evaluate(modelnum)
        obs = {}
        if record_sources:
            for rec in record_sources:
                v = self._extract_observable(modelnum, rec['source'])
                obs[rec['name']] = float(v) if (v is not None and not np.isnan(v)) else np.nan
        return chi2, obs

    def sample_flat(self) -> None:
        '''Function for sampling parameters defined in config_dict ranges.'''

        self.points.update(sampling.sample_flat(self.parameters, self.num_models))

        return None

    def sample_fixed(self) -> None:
        '''Function for generating fixed points given by input config file.'''

        self.points.update(sampling.sample_fixed(self.parameters))

        return None

    def _validate_observable_sources(self) -> None:
        '''For prior: mcmc, every constraint / record_observables source.step must be a
        step that runs (else _extract_observable returns None for every point and the
        chain never moves), and every constraint uncertainty must be > 0.'''
        if getattr(self, 'prior', None) != 'mcmc':
            return
        step_names = {s['name'] for s in self.steps}
        srcs = [c.get('source', {}) for c in (getattr(self, 'constraints', []) or [])]
        srcs += [r.get('source', {}) for r in ((getattr(self, 'mcmc', {}) or {}).get('record_observables', []) or [])]
        missing = sorted({s.get('step') for s in srcs if s.get('step') and s.get('step') not in step_names})
        if missing:
            raise ValueError(
                f"config references observable source step(s) {missing} not in the pipeline "
                f"steps {sorted(step_names)}; every constraint/record_observables source.step "
                f"must be a step that runs.")
        for c in (getattr(self, 'constraints', []) or []):
            # hard_window is a veto (0 inside / inf outside): it has no central value and
            # divides by nothing, so it needs lo and/or hi instead of an uncertainty.
            if c.get('type') == 'hard_window':
                if c.get('lo') is None and c.get('hi') is None:
                    raise ValueError(f"constraint '{c.get('name')}' is type hard_window but "
                                     f"sets neither 'lo' nor 'hi'; it would veto nothing.")
                lo, hi = c.get('lo'), c.get('hi')
                if lo is not None and hi is not None and float(lo) >= float(hi):
                    raise ValueError(f"constraint '{c.get('name')}' has empty hard_window "
                                     f"lo={lo!r} >= hi={hi!r}; every point would be vetoed.")
                continue
            if float(c.get('uncertainty', 0)) <= 0:
                raise ValueError(f"constraint '{c.get('name')}' has non-positive uncertainty "
                                 f"{c.get('uncertainty')!r}; it must be > 0 (chi2 divides by it).")
        # record_observables ntuple branch names must be unique and must not shadow the
        # reserved emcee branches (emcee_logp / chi2); caught here so it fails at config
        # load rather than after the whole scan, when the ntuple is written. (IN_* vs
        # record collisions are caught in emcee_ntuple.build_ntuple_tree.)
        reserved = {'emcee_logp', 'chi2'}
        branches = [r.get('branch') for r in
                    ((getattr(self, 'mcmc', {}) or {}).get('record_observables', []) or [])
                    if r.get('branch')]
        dupes = sorted({b for b in branches if branches.count(b) > 1})
        clash = sorted(reserved.intersection(branches))
        if dupes or clash:
            raise ValueError(
                f"invalid record_observables branch names: "
                + (f"duplicates {dupes}; " if dupes else "")
                + (f"reserved {clash}; " if clash else "")
                + f"each 'branch' must be unique and not one of {sorted(reserved)}.")

    def _run_and_evaluate(self, modelnum: int) -> float:
        '''Run the pipeline for modelnum and return chi2 from the constraints.

        Returns inf when a CONSTRAINT observable cannot be evaluated ("this point failed /
        is unusable"); a genuinely computed but very bad point keeps its large FINITE chi2
        (the caller tells the two apart by finiteness). Steps run in order; a step whose
        input file is absent (an upstream dependency failed) is skipped, so a failure in a
        step NOT needed by any constraint does not veto the point.'''
        for step in self.steps:
            indir = step.get('input_dir')
            if indir and not os.path.exists(f"{self.scan_dir}/{indir}/{modelnum}.slha"):
                continue  # an upstream step produced no output -> this one cannot run
            self.run_step(modelnum=modelnum, **step)

        chi2 = 0.0
        for constraint in self.constraints:
            obs_value = self._extract_observable(modelnum, constraint['source'])
            if obs_value is None or not np.isfinite(obs_value):
                return float('inf')  # a constraint observable is unavailable -> unusable point
            chi2 += self._compute_chi2_term(obs_value, constraint)
        return chi2

    def _extract_observable(self, modelnum: int, source: dict) -> float:
        '''Extract an observable value from a pipeline output file.

        source dict keys:
          step      : pipeline step name (e.g. "SPheno", "micromegas")
          file_type : "slha", "csv", or "flha"
          block     : SLHA/FLHA block name (for slha/flha types)
          entry     : entry key, int or list for tuple keys (for slha/flha types)
          variable  : variable name (for csv type)
        '''
        # Find the output directory for this step
        step_config = None
        for s in self.steps:
            if s['name'] == source['step']:
                step_config = s
                break
        if step_config is None:
            log.error(f"Step {source['step']} not found in pipeline steps")
            return None

        file_type = source['file_type']

        try:
            if file_type in ('slha', 'flha'):
                ext = '.slha' if file_type == 'slha' else '.flha'
                filepath = f"{self.scan_dir}/{step_config['output_dir']}/{modelnum}{ext}"
                slha_data = pyslha.read(filepath, ignorenomass=True, ignoreblocks=['SPINFO'])
                entry_key = source['entry']
                if isinstance(entry_key, list):
                    entry_key = tuple(entry_key)
                value = slha_data.blocks[source['block']].entries[entry_key]
                if isinstance(value, tuple):
                    value = value[0]
                return float(value)

            elif file_type == 'csv':
                filepath = f"{self.scan_dir}/{step_config['output_dir']}/{modelnum}.csv"
                # Invalidate csvReader cache so we always read fresh data
                csvReader.cacheName = ""
                csv_data = csvReader.read(filepath)
                variable = source['variable']
                for col in csv_data:
                    if variable in csv_data[col]:
                        return float(csv_data[col][variable])
                # Suppressed under quiet (emcee engine): invalid points fail
                # extraction on every walker eval -> log spam.
                if not self.quiet:
                    log.warning(f"Variable {variable} not found in {filepath}")
                return None

            else:
                log.error(f"Unsupported file_type: {file_type}")
                return None

        except Exception as e:
            # Suppressed under quiet (emcee engine): invalid spectra (e.g. no MASS
            # block) fail extraction on every walker eval -> log spam.
            if not self.quiet:
                log.warning(f"Failed to extract observable from {source}: {e}")
            return None

    @staticmethod
    def _compute_chi2_term(obs_value: float, constraint: dict) -> float:
        '''Compute chi-squared contribution from a single constraint.

        `hard_window` is a VETO, not a penalty: it contributes 0 inside the window and
        inf outside, so _run_and_evaluate's finiteness check drops the point exactly as
        it drops a failed pipeline step. Bounds are strict (lo < value < hi) and each is
        optional, which expresses the three shapes a hard requirement takes: a floor
        (`lo` only, e.g. m_h > 0 to reject SPheno's -1 sentinel), a window (`lo` + `hi`,
        e.g. 0 < Omega < 1 for "does not overclose the universe"), and a boolean flag
        (`lo: 0.5` on a 0/1 observable such as micromegas' cdm_is_neutralino).
        '''
        ctype = constraint.get('type', 'gaussian')

        if ctype == 'hard_window':
            lo, hi = constraint.get('lo'), constraint.get('hi')
            if lo is not None and not obs_value > float(lo):
                return float('inf')
            if hi is not None and not obs_value < float(hi):
                return float('inf')
            return 0.0

        central = constraint['central']
        sigma = constraint['uncertainty']

        if ctype == 'gaussian':
            return ((obs_value - central) / sigma) ** 2
        elif ctype == 'upper_bound':
            if obs_value > central:
                return ((obs_value - central) / sigma) ** 2
            return 0.0
        elif ctype == 'lower_bound':
            if obs_value < central:
                return ((central - obs_value) / sigma) ** 2
            return 0.0
        else:
            raise ValueError(f"Unknown constraint type: {ctype}")

    # ── Grid sampling ─────────────────────────────────────────────────

    def sample_grid(self) -> None:
        """Generate models on a Cartesian grid over specified parameters.

        Parameters in self.parameters are split into two categories based on
        the YAML config field ``grid`` (dict mapping param name → list of
        explicit grid values).  Parameters listed in ``grid`` are expanded
        into a Cartesian product; all other parameters are sampled uniformly
        at random (one draw per grid point) from their ranges defined in
        ``parameters``.

        After expansion, ``self.num_models`` is overridden with the grid size
        and ``self.points`` is populated identically to ``sample_flat``.
        """
        points, n_grid = sampling.sample_grid(self.parameters, getattr(self, 'grid', {}))

        # Override num_models
        self.num_models = n_grid
        self.points.update(points)

    # ── End of grid sampling ──────────────────────────────────────────

    # ── Importance sampling ──────────────────────────────────────────

    def sample_importance(self) -> None:
        """Generate models by perturbing known viable seed parameters.

        Reads seed CSV from self.importance['seed_file'], applies Gaussian
        perturbations with widths from self.importance['proposal_widths'],
        and generates perturbations_per_seed variants of each seed.
        Uses _propose() for boundary-aware perturbation.
        """
        points, n_total = sampling.sample_importance(self.parameters, getattr(self, 'importance', {}))

        # Override num_models
        self.num_models = n_total
        self.points.update(points)

    # ── End of importance sampling ────────────────────────────────────

    def run_step(self, modelnum: int, **kwargs) -> bool:
        '''Function decorator for exectuting run_* function. Returns success bool.

        This is the GATE: it reports whether the tool ran, and drives which models the
        next step attempts. It deliberately does NOT fold in whether the output is
        ntuple-usable — see :meth:`step_produced_ntuple_vars` for that, which is
        reported separately so a stricter statistic cannot silently shrink the
        dataset.
        '''
        step_name = kwargs['name']

        # Select run_ function based on step name
        if hasattr(self, f"run_{step_name}") and callable(func := getattr(self, f"run_{step_name}")):
            return func(modelnum, **kwargs)
        else:
            raise KeyError(f"step {step_name} not supported!")

    def step_produced_ntuple_vars(self, modelnum: int, step: dict) -> bool:
        '''Did `step` leave variables the ntuple can carry for `modelnum`?

        The honest per-step success statistic. A tool's exit status is not evidence of
        usable output: SPheno's "charged LSP" SPINFO warning makes pyslha raise on an
        otherwise fine spectrum, and SModelS writes a result file with no ExptRes at
        all when it was given no cross sections — a 200-model scan reported
        "SModelS 116/116 succeeded" while its ntuple got zero SModelS branches. This
        counts the models that actually contribute, using the same extraction the
        ntuple performs.
        '''
        return ntupling.step_produced_ntuple_vars(step, modelnum, self.scan_dir,
                                                  getattr(self, 'isGMSB', False))
    
    def run_prep_input(self, modelnum: int, **kwargs) -> bool:
        '''Prep input. Returns the filename of the prepped file.'''

        point = {key: self.points[key][modelnum] for key in self.points}
        preppedfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"

        return toolsteps.run_prep_input(point=point, rawfilen=self.rawfilen, outfile=preppedfile)

    def run_SPheno(self, modelnum: int, **kwargs) -> bool:
        '''Run SPheno and return True if successful. Re-add input blocks to output file if successful. Relevant for running tools on output file.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        logfile = f"{self.scan_dir}/{kwargs['log_dir']}/{modelnum}.log"

        return toolsteps.run_SPheno(infile=infile, outfile=outfile, logfile=logfile, rawfilen=self.rawfilen, timeout=self.tool_timeout)

    def run_softsusy(self, modelnum: int, **kwargs) -> bool:
        '''Run softsusy and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"

        return toolsteps.run_softsusy(infile=infile, outfile=outfile, timeout=self.tool_timeout)

    def run_micromegas(self, modelnum: int, **kwargs) -> bool:
        '''Run micromegas and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile_raw = f"{self.scan_dir}/{kwargs['output_dir']}_raw/{modelnum}.out"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.csv"

        return toolsteps.run_micromegas(infile=infile, outfile_raw=outfile_raw, outfile=outfile, timeout=self.tool_timeout)

    def run_superiso(self, modelnum: int, **kwargs) -> bool:
        '''Run superiso and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.flha"

        return toolsteps.run_superiso(infile=infile, outfile=outfile, timeout=self.tool_timeout)

    def run_feynhiggs(self, modelnum: int, **kwargs) -> bool:
        '''Run FeynHiggs and return True if successful. Re-add input blocks to output file if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        logfile = f"{self.scan_dir}/{kwargs['log_dir']}/{modelnum}.log"

        return toolsteps.run_feynhiggs(infile=infile, outfile=outfile, logfile=logfile, rawfilen=self.rawfilen, timeout=self.tool_timeout)

    def run_gm2calc(self, modelnum: int, **kwargs) -> bool:
        '''Run GM2Calc and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"

        return toolsteps.run_gm2calc(infile=infile, outfile=outfile, timeout=self.tool_timeout)

    def run_evade(self, modelnum: int, **kwargs) -> bool:
        '''Run EVADE and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.tsv"
        evade_conffile = f"{self.scan_dir}/{kwargs['output_dir']}_config/{modelnum}.cfg"
        evade_infile = f"{self.scan_dir}/{kwargs['output_dir']}_evade_in/{modelnum}.in"

        return toolsteps.run_evade(infile=infile, outfile=outfile, evade_conffile=evade_conffile, evade_infile=evade_infile, timeout=self.tool_timeout)

    def run_SModelS(self, modelnum: int, **kwargs) -> bool:
        '''Run SModelS and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outputDir = f"{self.scan_dir}/{kwargs['output_dir']}/"
        smodels_config = f"{datadir}/SModelS_parameters.ini"

        # xsec_dest=None: legacy in-place xsec injection into the SPheno output —
        # correct here because the sequential per-model loop has already run every
        # other SPheno consumer for this model.
        return toolsteps.run_SModelS(infile=infile, output_dir=outputDir, smodels_ini=smodels_config, xsec_dest=None)
    
    def generate_models(self) -> None:
        '''Main function to generate models.'''
        
        log.info(f"Starting Model Generation...")
        
        # Set up directories for saving scan. The destructive wipe is a legacy-CLI
        # behavior (a fresh scan overwrites the scan dir); the Snakemake rules own
        # their outputs incrementally and call _setup_scan_dirs() directly instead.
        log.info(f"Setting up output directories for scan")
        pipeline.wipe_scan_dir(self.scan_dir)
        self._setup_scan_dirs()
        
        if self.prior == 'flat':
            self.sample_flat()
        elif self.prior == 'fixed':
            self.sample_fixed()
        elif self.prior == 'mcmc':
            # prior: mcmc is the emcee sampler, which runs the pipeline inside sampling
            # and owns its chain.h5 checkpoint/resume. It is driven by the `chain`
            # Snakemake rule (r3mg_batch.py chain -> run_emcee_scan), NOT generate_models.
            raise ValueError(
                "prior: mcmc runs the emcee sampler via the 'chain' rule "
                "(r3mg_batch.py chain), not generate_models(); use genModels.py.")
        elif self.prior == 'grid':
            self.sample_grid()
        elif self.prior == 'importance':
            self.sample_importance()
        else:
            raise ValueError(f"ERROR: prior {self.prior} not supported. Can be: flat, fixed, mcmc, grid, importance.")

        # prep dicts to keep track of successes. `succcounts` is the gate (the tool
        # ran); `usablecounts` is the honest statistic (the output yields ntuple
        # variables). They differ exactly where a tool exits cleanly having produced
        # nothing the ntuple can carry, which is what the stats line must not hide.
        succcounts = {}
        usablecounts = {}
        for step in self.steps:
            succcounts[step['name']] = 0
            usablecounts[step['name']] = 0

        # Gate on the step that PRODUCES this step's input_dir (mirrors
        # workflow/rules/generate.smk), not on the previous entry in the config list:
        # micromegas, superiso, gm2calc, feynhiggs and SModelS all read the SAME
        # SPheno output, so one of them saying nothing must not stop the others from
        # being attempted.
        producer_of = {s['output_dir']: s['name'] for s in self.steps if s.get('output_dir')}

        for mod in range(self.num_models):
            log.info(f"Generating Model: {mod}.slha ({mod+1}/{self.num_models})")

            ran_ok = set()
            for step in self.steps:
                indir = step.get('input_dir')
                gate = producer_of.get(indir) if indir else None
                if gate is not None and gate not in ran_ok:
                    log.info(f"\tskipping step: {step['name']} (its input step {gate} failed)")
                    continue
                log.info(f"\trunning step: {step['name']}")
                success = self.run_step(modelnum=mod, **step)

                # Keep track of successes
                if success:
                    ran_ok.add(step['name'])
                    succcounts[step['name']] += 1
                    if self.step_produced_ntuple_vars(mod, step):
                        usablecounts[step['name']] += 1

        self._dump_scan_config()

        log.info("Finished generating models! Stats per step "
                 "(with ntuple variables / tool ran / attempted):")
        for step in succcounts:
            usable, ran = usablecounts[step], succcounts[step]
            note = "" if usable == ran else \
                f"   <- {ran - usable} ran but produced NO ntuple variables"
            log.info(f"\t{step}: {usable}/{ran}/{self.num_models}.{note}")

        return None

    def _dump_scan_config(self) -> None:
        '''Dump scan configuration to YAML file in scan directory.'''
        with open(f"{self.scan_dir}/scan_config.yaml", "w") as file:
            # 'quiet' is a logging knob, not scan configuration — keep the
            # dumped yaml identical to the legacy format
            skipkeys = ['points', 'quiet']
            dumpdict = {key: value for key, value in vars(self).items() if key not in skipkeys}
            yaml.dump(dumpdict, file, default_flow_style=None, encoding=None)
    
    def mkntuple(self) -> None:
        '''Promote ntuple making to class attribute.'''
        
        return mkntuple(self.steps, self.scan_dir, self.num_models, self.isGMSB)