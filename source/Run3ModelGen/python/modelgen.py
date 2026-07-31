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
import shutil

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
        self.scan_dir = scan_dir
        self.rawfilen = f"{datadir}/raw.slha"
        self.seed = seed
        self.points = {}
        
        # Add all keys in config_file as attributes. Replace attribute by default if not
        # provided; extra keys not in the default (e.g. 'grid') are carried through.
        default_config = f"{datadir}/configs/default_config.yaml"
        for key, value in pipeline.load_scan_config(self.config_file, default_config).items():
            setattr(self, key, value)

        # Drop pipeline steps whose external tool is not installed on this platform
        # (e.g. FeynHiggs is not packaged on conda-forge for macOS).
        self.steps = pipeline.resolve_steps(getattr(self, 'steps', []))

        # Check if too many models supplied
        pipeline.check_model_cap(self.num_models)
        
        # Set numpy seed:
        np.random.seed(self.seed)
        
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

    def sample_flat(self) -> None:
        '''Function for sampling parameters defined in config_dict ranges.'''

        self.points.update(sampling.sample_flat(self.parameters, self.num_models))

        return None

    def sample_fixed(self) -> None:
        '''Function for generating fixed points given by input config file.'''

        self.points.update(sampling.sample_fixed(self.parameters))

        return None

    # ── MCMC sampling methods ──────────────────────────────────────────

    def sample_mcmc(self) -> None:
        '''Run Metropolis-Hastings MCMC sampling.

        Unlike sample_flat(), this method also runs the pipeline for each proposal,
        since the likelihood depends on computed observables. After completion,
        self.points contains only accepted models and output files exist at indices
        0 through num_models-1.

        Diagnostics:
            - Per-window log lines include the Metropolis acceptance rate
              (`accept_metro`), which is what determines mixing — independent
              of how many post-burn-in samples have been *committed*.
            - End-of-run `[mcmc-summary]` line is emitted in a parseable format
              for cross-chain diagnostics (split-R̂, ESS_bulk via
              `scripts/mcmc_diagnostics.py`).

        Optional adaptive proposal scaling during burn-in:
            mcmc:
              adapt_proposals:
                enabled: true
                target_acceptance: 0.25   # Roberts-Gelman-Gilks high-D optimum
                window: 200               # proposals between adaptations
                learning_rate: 0.05       # multiplicative step (log-scale)
        '''
        proposal_widths = self._build_proposal_widths()
        burn_in = self.mcmc.get('burn_in', 0)
        max_attempts = self.mcmc.get('max_attempts', 100000)
        accept_invalid = self.mcmc.get('accept_invalid', False)
        restart_after = self.mcmc.get('restart_after', None)
        penalty = 1e6

        # Adaptive proposal scaling (only during burn-in, to preserve detailed
        # balance once we start collecting samples).
        adapt_cfg = self.mcmc.get('adapt_proposals', {}) or {}
        adapt_enabled = bool(adapt_cfg.get('enabled', False))
        adapt_target = float(adapt_cfg.get('target_acceptance', 0.25))
        adapt_window = int(adapt_cfg.get('window', 200))
        adapt_lr = float(adapt_cfg.get('learning_rate', 0.05))

        # Use num_models as a scratch file index, separate from any valid
        # accepted index (0..num_models-1). This way every acceptance copies
        # from scratch to the final index, and no accepted files get overwritten.
        scratch_idx = self.num_models

        # Initialize self.points large enough to index scratch_idx
        # (run_prep_input reads self.points[key][modelnum])
        self._init_points_arrays(scratch_idx + 1)

        # Draw random initial point within parameter bounds
        current = self._sample_initial_point()
        for key, val in current.items():
            self.points[key][scratch_idx] = val

        # Run pipeline on initial point at scratch index
        current_chi2 = self._run_and_evaluate(modelnum=scratch_idx)

        accepted_points = {key: [] for key in self.parameters}
        n_accepted = 0          # committed (post-burn-in, valid) samples
        n_total = 0              # total proposals attempted
        n_post_burnin = 0        # proposals attempted after burn-in
        n_burn_accepted = 0
        n_restarts = 0
        n_since_restart = 0
        # New diagnostic counters: Metropolis-rule acceptance and pipeline
        # validity. These are independent of `n_accepted` (which only counts
        # post-burn-in valid commits), and they're what mixing actually depends on.
        n_metro_accepted = 0     # times the MH rule fired (any phase)
        n_pipeline_invalid = 0   # proposals where the pipeline failed
        n_window_proposed = 0    # rolling counts for adaptive scaling
        n_window_accepted = 0
        past_burn_in = (burn_in == 0)
        if not past_burn_in:
            log.info(f"Starting MCMC burn-in ({burn_in} accepted steps)...")
        if adapt_enabled:
            log.info(f"Adaptive proposal scaling enabled "
                     f"(target_acceptance={adapt_target:.2f}, "
                     f"window={adapt_window}, learning_rate={adapt_lr:.3f})")

        while n_accepted < self.num_models and n_total < max_attempts:
            n_total += 1
            n_since_restart += 1
            n_window_proposed += 1

            # Restart if burn-in is not converging
            if restart_after is not None and not past_burn_in and n_since_restart >= restart_after:
                n_restarts += 1
                log.info(f"MCMC restart #{n_restarts} after {n_since_restart} attempts "
                         f"({n_burn_accepted}/{burn_in} burn-in steps accepted)")
                current = self._sample_initial_point()
                for key, val in current.items():
                    self.points[key][scratch_idx] = val
                current_chi2 = self._run_and_evaluate(modelnum=scratch_idx)
                n_burn_accepted = 0
                n_since_restart = 0
                # Reset adaptation window so a freshly-restarted chain re-tunes.
                n_window_proposed = 0
                n_window_accepted = 0

            proposed = self._propose(current, proposal_widths)

            # Write proposed point into scratch arrays at scratch index
            for key, val in proposed.items():
                self.points[key][scratch_idx] = val

            # Run pipeline at scratch index
            proposed_chi2 = self._run_and_evaluate(modelnum=scratch_idx)
            if proposed_chi2 >= penalty:
                n_pipeline_invalid += 1

            if self._accept_proposal(current_chi2, proposed_chi2):
                n_metro_accepted += 1
                n_window_accepted += 1
                current = proposed.copy()
                current_chi2 = proposed_chi2

                # Skip saving invalid points unless accept_invalid is set
                if not accept_invalid and current_chi2 >= penalty:
                    pass  # state moved (detailed balance), but don't commit
                elif past_burn_in:
                    self._commit_accepted_model(scratch_idx, n_accepted)
                    for key in self.parameters:
                        accepted_points[key].append(current[key])
                    n_accepted += 1
                    accepted_file = f"{self.scan_dir}/input/{n_accepted - 1}.slha"
                    log.info(f"MCMC accepted {n_accepted}/{self.num_models} "
                             f"(attempt {n_total}, file={accepted_file}, chi2={current_chi2:.2f})")
                else:
                    n_burn_accepted += 1
                    if n_burn_accepted >= burn_in:
                        past_burn_in = True
                        log.info(f"MCMC burn-in complete after {n_burn_accepted} accepted steps")

            # Adaptive proposal scaling (burn-in only). Multiplicatively scale
            # all widths to drive the rolling window's acceptance toward
            # `adapt_target`. Detailed balance is preserved because adaptation
            # stops the moment burn-in ends.
            if (adapt_enabled and not past_burn_in
                    and n_window_proposed >= adapt_window):
                window_rate = n_window_accepted / n_window_proposed
                # Avoid divide-by-zero if user sets target_acceptance=0
                target = max(adapt_target, 1e-3)
                # Robbins-Monro-style log-scale step toward target
                factor = math.exp(adapt_lr * (window_rate - target) / target)
                # Cap factor to keep adaptation gentle and avoid runaway scaling
                factor = max(0.5, min(2.0, factor))
                for k in proposal_widths:
                    proposal_widths[k] *= factor
                log.info(f"[adapt] window accept_rate={window_rate:.1%}, "
                         f"factor={factor:.3f}; current widths={proposal_widths}")
                n_window_proposed = 0
                n_window_accepted = 0

            # Periodic progress logging
            if past_burn_in:
                n_post_burnin += 1
            if n_total % 100 == 0:
                accept_metro = n_metro_accepted / max(n_total, 1) * 100
                invalid_frac = n_pipeline_invalid / max(n_total, 1) * 100
                if past_burn_in:
                    commit_rate = (n_accepted / n_post_burnin * 100) if n_post_burnin > 0 else 0
                    log.info(f"MCMC progress: committed={n_accepted}/{self.num_models}, "
                             f"total={n_total}, "
                             f"accept_metro={accept_metro:.1f}%, "
                             f"commit_rate={commit_rate:.1f}%, "
                             f"invalid={invalid_frac:.1f}%")
                else:
                    log.info(f"MCMC burn-in progress: burn_accepted={n_burn_accepted}/{burn_in}, "
                             f"total={n_total}, "
                             f"accept_metro={accept_metro:.1f}%, "
                             f"invalid={invalid_frac:.1f}%")

        # Clean up scratch files
        self._cleanup_scratch_files(scratch_idx)

        # Replace self.points with accepted values
        for key in self.parameters:
            self.points[key] = np.array(accepted_points[key])

        if n_accepted < self.num_models:
            log.warning(f"MCMC stopped at max_attempts={max_attempts} with {n_accepted} models")
            self.num_models = n_accepted

        # Parseable end-of-run summary. Each chain emits one of these — across
        # chains, scripts/mcmc_diagnostics.py computes split-R̂ and ESS_bulk.
        log.info(f"MCMC complete: {n_accepted} models, {n_total} proposals, "
                 f"rate={n_accepted/max(n_total,1)*100:.1f}%")
        log.info(
            f"[mcmc-summary] proposed={n_total} "
            f"metro_accepted={n_metro_accepted} "
            f"metro_accept_rate={n_metro_accepted/max(n_total,1):.4f} "
            f"committed={n_accepted} "
            f"pipeline_invalid={n_pipeline_invalid} "
            f"invalid_rate={n_pipeline_invalid/max(n_total,1):.4f} "
            f"restarts={n_restarts} "
            f"final_widths={proposal_widths}"
        )
        if n_metro_accepted / max(n_total, 1) > 0.6:
            log.warning(
                "[mcmc-summary] Metropolis acceptance > 60% — proposal widths "
                "are too small. Mixing will be slow; consider widening or "
                "enabling adapt_proposals."
            )
        elif n_metro_accepted / max(n_total, 1) < 0.10:
            log.warning(
                "[mcmc-summary] Metropolis acceptance < 10% — proposal widths "
                "are too large. Most compute is wasted on rejections; consider "
                "narrowing or enabling adapt_proposals."
            )

        return None

    def _build_proposal_widths(self) -> dict:
        '''Build dict of proposal widths, using configured values or 5% of range as default.'''
        return sampling.build_proposal_widths(self.parameters, self.mcmc)

    def _sample_initial_point(self) -> dict:
        '''Sample a random starting point within all parameter bounds.'''
        return sampling.sample_initial_point(self.parameters)

    def _propose(self, current: dict, widths: dict) -> dict:
        '''Propose a new point using Gaussian perturbation with boundary handling.'''
        return sampling.propose(current, widths, self.parameters)

    @staticmethod
    def _reflect(value: float, low: float, high: float) -> float:
        '''Reflect value into [low, high] interval to preserve detailed balance.'''
        return sampling.reflect(value, low, high)

    def _run_and_evaluate(self, modelnum: int) -> float:
        '''Run all pipeline steps for modelnum, then compute chi2 from constraints.
        Returns a large penalty value if the pipeline fails.'''
        penalty = 1e6

        for step in self.steps:
            success = self.run_step(modelnum=modelnum, **step)
            if not success:
                return penalty

        chi2 = 0.0
        for constraint in self.constraints:
            obs_value = self._extract_observable(modelnum, constraint['source'])
            if obs_value is None or np.isnan(obs_value):
                return penalty
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
                log.warning(f"Variable {variable} not found in {filepath}")
                return None

            else:
                log.error(f"Unsupported file_type: {file_type}")
                return None

        except Exception as e:
            log.warning(f"Failed to extract observable from {source}: {e}")
            return None

    @staticmethod
    def _compute_chi2_term(obs_value: float, constraint: dict) -> float:
        '''Compute chi-squared contribution from a single constraint.'''
        central = constraint['central']
        sigma = constraint['uncertainty']
        ctype = constraint.get('type', 'gaussian')

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

    def _accept_proposal(self, current_chi2: float, proposed_chi2: float) -> bool:
        '''Metropolis-Hastings acceptance criterion.
        Likelihood is proportional to exp(-chi2/2).'''
        delta_chi2 = proposed_chi2 - current_chi2
        if delta_chi2 <= 0:
            return True
        return np.random.uniform() < np.exp(-delta_chi2 / 2.0)

    def _commit_accepted_model(self, scratch_idx: int, accepted_idx: int) -> None:
        '''Copy pipeline output files from scratch index to final accepted index.'''
        for step in self.steps:
            if 'output_dir' not in step:
                continue
            odir = f"{self.scan_dir}/{step['output_dir']}"

            # Determine file extension based on step type
            if 'micromegas' in step['name']:
                exts = ['.csv']
            elif 'superiso' in step['name']:
                exts = ['.flha']
            elif 'evade' in step['name']:
                exts = ['.tsv']
            elif 'SModelS' in step['name']:
                exts = ['.slha.py']
            else:
                exts = ['.slha']

            for ext in exts:
                src = f"{odir}/{scratch_idx}{ext}"
                dst = f"{odir}/{accepted_idx}{ext}"
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            # Handle extra directories for micromegas raw output
            if 'micromegas' in step['name']:
                raw_src = f"{odir}_raw/{scratch_idx}.out"
                raw_dst = f"{odir}_raw/{accepted_idx}.out"
                if os.path.exists(raw_src):
                    shutil.copy2(raw_src, raw_dst)

            # Handle log files
            if 'log_dir' in step:
                ldir = f"{self.scan_dir}/{step['log_dir']}"
                lsrc = f"{ldir}/{scratch_idx}.log"
                ldst = f"{ldir}/{accepted_idx}.log"
                if os.path.exists(lsrc):
                    shutil.copy2(lsrc, ldst)

    def _cleanup_scratch_files(self, scratch_idx: int) -> None:
        '''Remove scratch files left over from the MCMC working index.'''
        for step in self.steps:
            if 'output_dir' not in step:
                continue
            odir = f"{self.scan_dir}/{step['output_dir']}"

            if 'micromegas' in step['name']:
                exts = ['.csv']
            elif 'superiso' in step['name']:
                exts = ['.flha']
            elif 'evade' in step['name']:
                exts = ['.tsv']
            elif 'SModelS' in step['name']:
                exts = ['.slha.py']
            else:
                exts = ['.slha']

            for ext in exts:
                f = f"{odir}/{scratch_idx}{ext}"
                if os.path.exists(f):
                    os.remove(f)

            if 'micromegas' in step['name']:
                f = f"{odir}_raw/{scratch_idx}.out"
                if os.path.exists(f):
                    os.remove(f)

            if 'log_dir' in step:
                f = f"{self.scan_dir}/{step['log_dir']}/{scratch_idx}.log"
                if os.path.exists(f):
                    os.remove(f)

    # ── End of MCMC methods ───────────────────────────────────────────

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

        return toolsteps.run_SPheno(infile=infile, outfile=outfile, logfile=logfile, rawfilen=self.rawfilen)

    def run_softsusy(self, modelnum: int, **kwargs) -> bool:
        '''Run softsusy and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"

        return toolsteps.run_softsusy(infile=infile, outfile=outfile)

    def run_micromegas(self, modelnum: int, **kwargs) -> bool:
        '''Run micromegas and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile_raw = f"{self.scan_dir}/{kwargs['output_dir']}_raw/{modelnum}.out"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.csv"

        return toolsteps.run_micromegas(infile=infile, outfile_raw=outfile_raw, outfile=outfile)

    def run_superiso(self, modelnum: int, **kwargs) -> bool:
        '''Run superiso and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.flha"

        return toolsteps.run_superiso(infile=infile, outfile=outfile)

    def run_feynhiggs(self, modelnum: int, **kwargs) -> bool:
        '''Run FeynHiggs and return True if successful. Re-add input blocks to output file if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        logfile = f"{self.scan_dir}/{kwargs['log_dir']}/{modelnum}.log"

        return toolsteps.run_feynhiggs(infile=infile, outfile=outfile, logfile=logfile, rawfilen=self.rawfilen)

    def run_gm2calc(self, modelnum: int, **kwargs) -> bool:
        '''Run GM2Calc and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"

        return toolsteps.run_gm2calc(infile=infile, outfile=outfile)

    def run_evade(self, modelnum: int, **kwargs) -> bool:
        '''Run EVADE and return True if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.tsv"
        evade_conffile = f"{self.scan_dir}/{kwargs['output_dir']}_config/{modelnum}.cfg"
        evade_infile = f"{self.scan_dir}/{kwargs['output_dir']}_evade_in/{modelnum}.in"

        return toolsteps.run_evade(infile=infile, outfile=outfile, evade_conffile=evade_conffile, evade_infile=evade_infile)

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
        os.system(f"rm -r {self.scan_dir} &>/dev/null")
        self._setup_scan_dirs()
        
        if self.prior == 'flat':
            self.sample_flat()
        elif self.prior == 'fixed':
            self.sample_fixed()
        elif self.prior == 'mcmc':
            # MCMC runs the pipeline internally during sampling.
            # After sample_mcmc() returns, self.points is populated with accepted models,
            # and all output files are already written at indices 0..num_models-1.
            self.sample_mcmc()
            self._dump_scan_config()
            return None
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