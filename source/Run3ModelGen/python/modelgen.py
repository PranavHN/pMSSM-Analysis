'''Module for ModelGenerator class'''

import yaml
import numpy as np
import os
datadir = os.environ['DATAPATH'].split(':')[0]

import pyslha

from Run3ModelGen.ntupling import mkntuple
from Run3ModelGen.microextract import microextract
from Run3ModelGen.addinputblocks import addinputblocks
from Run3ModelGen.pMSSM_convert import convert_slha
import Run3ModelGen.csvReader as csvReader
import shutil

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)  # so we have logger names

# Bridge handler: this forwards standard log records → structlog
class StructlogHandler(logging.Handler):
    def emit(self, record):
        # Convert logging.LogRecord → structlog event
        log.bind(module=record.name).info(record.getMessage())

class ModelGenerator:
    '''Class for Model Generation.'''
    def __init__(self, config_file: str = None, scan_dir: str = f"scan", seed: int = 123) -> None:
        '''Initialise scan.'''
        
        # Print logo. Note: ASCII art generated with https://patorjk.com/software/taag/ (Small, Fitted)
        with open(f"{datadir}/logo.txt", 'r') as file:
            logo = "\n"+file.read()+"\n"
        print(logo)
        
        self.config_file = config_file
        self.scan_dir = scan_dir
        self.rawfilen = f"{datadir}/raw.slha"
        self.seed = seed
        self.points = {}
        
        # Add all keys in config_file as attributes. Replace attribute by default if not provided
        default_config = f"{datadir}/configs/default_config.yaml"
        config_dict = self.read_yaml_file(self.config_file) if self.config_file is not None else {}
        
        for key, defaultval in self.read_yaml_file(default_config).items():
            if key in config_dict:
                setattr(self, key, config_dict[key])
            # If num_models is not supplied and the prior is 'fixed', read number of models from parameters
            elif key == 'num_models' and config_dict.get('prior', None) == 'fixed':
                setattr(self, key, len(config_dict['parameters']['tanb']))
            else:
                setattr(self, key, defaultval)

        # Load extra config keys not in default (e.g., 'grid' for grid sampling)
        default_keys = set(self.read_yaml_file(default_config).keys())
        for key, value in config_dict.items():
            if key not in default_keys:
                setattr(self, key, value)
        
        # Check if too many models supplied
        if self.num_models > 10000:
            log.fatal(f"num_models too large! Please reduce to max 10,000 and split into multiple jobs using genModels_condor.py!")
            exit()
        
        # Set numpy seed:
        np.random.seed(self.seed)
        
        # Print information on initialised generator
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
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        return data
    
    def sample_flat(self) -> None:
        '''Function for sampling parameters defined in config_dict ranges.'''
        
        for key, scanrange in self.parameters.items():
            
            # Check if the provided range is interrupted
            if type(scanrange[0]) == list:
                weights = []
                dist = 0
                
                # Loop over ranges to get weights for numbers of models
                for subrange in scanrange:
                    if type(subrange) != list: raise ValueError(f"Provided scanrange for parameter {key} is not supported!")
                    weights.append((subrange[1]-subrange[0]))
                    dist += subrange[1]-subrange[0]

                weights = [weight/dist for weight in weights]
                
                # Generate random points according to weights and append result in one np array
                # Making sure that overall number of models matches
                points_left = self.num_models
                points = np.array([])
                list_numpoints = []
                for i in range(len(scanrange)):
                    numpoints = int(self.num_models*weights[i])
                    if i == len(scanrange)-1: numpoints = points_left
                    list_numpoints += [numpoints]
                    
                    points = np.append(points, np.random.uniform(scanrange[i][0], scanrange[i][1], numpoints))
                    points_left -= numpoints
                    
                self.points[key] = points
                
                log.info(f"Found gap in scanning ranges for parameter {key}. Generated random numbers for subranges with weights:", weights=weights)
                log.info("Resulting in these numbers of points:", list_numpoints=list_numpoints)
                
            else:
                self.points[key] = np.random.uniform(scanrange[0], scanrange[1], self.num_models)
            
        return None
    
    def sample_fixed(self) -> None:
        '''Function for generating fixed points given by input config file.'''

        for key, pointlist in self.parameters.items():
            self.points[key] = np.array(pointlist)

        return None

    # ── MCMC sampling methods ──────────────────────────────────────────

    def sample_mcmc(self) -> None:
        '''Run Metropolis-Hastings MCMC sampling.

        Unlike sample_flat(), this method also runs the pipeline for each proposal,
        since the likelihood depends on computed observables. After completion,
        self.points contains only accepted models and output files exist at indices
        0 through num_models-1.
        '''
        proposal_widths = self._build_proposal_widths()
        burn_in = self.mcmc.get('burn_in', 0)
        max_attempts = self.mcmc.get('max_attempts', 100000)
        accept_invalid = self.mcmc.get('accept_invalid', False)
        penalty = 1e6

        # Use num_models as a scratch file index, separate from any valid
        # accepted index (0..num_models-1). This way every acceptance copies
        # from scratch to the final index, and no accepted files get overwritten.
        scratch_idx = self.num_models

        # Initialize self.points large enough to index scratch_idx
        # (run_prep_input reads self.points[key][modelnum])
        for key in self.parameters:
            self.points[key] = np.zeros(scratch_idx + 1)

        # Draw random initial point within parameter bounds
        current = self._sample_initial_point()
        for key, val in current.items():
            self.points[key][scratch_idx] = val

        # Run pipeline on initial point at scratch index
        current_chi2 = self._run_and_evaluate(modelnum=scratch_idx)

        accepted_points = {key: [] for key in self.parameters}
        n_accepted = 0
        n_total = 0
        n_burn_accepted = 0
        past_burn_in = (burn_in == 0)
        if not past_burn_in:
            log.info(f"Starting MCMC burn-in ({burn_in} accepted steps)...")

        while n_accepted < self.num_models and n_total < max_attempts:
            n_total += 1

            proposed = self._propose(current, proposal_widths)

            # Write proposed point into scratch arrays at scratch index
            for key, val in proposed.items():
                self.points[key][scratch_idx] = val

            # Run pipeline at scratch index
            proposed_chi2 = self._run_and_evaluate(modelnum=scratch_idx)

            if self._accept_proposal(current_chi2, proposed_chi2):
                current = proposed.copy()
                current_chi2 = proposed_chi2

                # Skip saving invalid points unless accept_invalid is set
                if not accept_invalid and current_chi2 >= penalty:
                    continue

                if past_burn_in:
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

            # Periodic progress logging
            if n_total % 100 == 0:
                rate = (n_accepted / n_total * 100) if n_total > 0 else 0
                log.info(f"MCMC progress: {n_accepted}/{self.num_models} accepted, "
                         f"{n_total} total attempts, acceptance rate: {rate:.1f}%")

        # Clean up scratch files
        self._cleanup_scratch_files(scratch_idx)

        # Replace self.points with accepted values
        for key in self.parameters:
            self.points[key] = np.array(accepted_points[key])

        if n_accepted < self.num_models:
            log.warning(f"MCMC stopped at max_attempts={max_attempts} with {n_accepted} models")
            self.num_models = n_accepted

        log.info(f"MCMC complete: {n_accepted} models, {n_total} proposals, "
                 f"rate={n_accepted/max(n_total,1)*100:.1f}%")

        return None

    def _build_proposal_widths(self) -> dict:
        '''Build dict of proposal widths, using configured values or 5% of range as default.'''
        configured = self.mcmc.get('proposal_widths', {})
        widths = {}
        for key, scanrange in self.parameters.items():
            if key in configured:
                widths[key] = configured[key]
            else:
                if isinstance(scanrange[0], list):
                    total_range = sum(sr[1] - sr[0] for sr in scanrange)
                else:
                    total_range = scanrange[1] - scanrange[0]
                widths[key] = abs(total_range) * 0.05
        return widths

    def _sample_initial_point(self) -> dict:
        '''Sample a random starting point within all parameter bounds.'''
        point = {}
        for key, scanrange in self.parameters.items():
            if isinstance(scanrange[0], list):
                # Pick a random subrange weighted by width, then uniform within it
                range_widths = [sr[1] - sr[0] for sr in scanrange]
                total = sum(range_widths)
                probs = [w / total for w in range_widths]
                idx = np.random.choice(len(scanrange), p=probs)
                point[key] = np.random.uniform(scanrange[idx][0], scanrange[idx][1])
            else:
                point[key] = np.random.uniform(scanrange[0], scanrange[1])
        return point

    def _propose(self, current: dict, widths: dict) -> dict:
        '''Propose a new point using Gaussian perturbation with boundary handling.'''
        proposed = {}
        for key, value in current.items():
            if widths[key] == 0:
                proposed[key] = value
                continue

            new_val = value + np.random.normal(0, widths[key])
            scanrange = self.parameters[key]

            if isinstance(scanrange[0], list):
                # Interrupted range: reject this dimension if proposed falls in a gap
                in_range = any(sr[0] <= new_val <= sr[1] for sr in scanrange)
                proposed[key] = new_val if in_range else value
            else:
                proposed[key] = self._reflect(new_val, scanrange[0], scanrange[1])

        return proposed

    @staticmethod
    def _reflect(value: float, low: float, high: float) -> float:
        '''Reflect value into [low, high] interval to preserve detailed balance.'''
        if low == high:
            return low
        while value < low or value > high:
            if value < low:
                value = 2 * low - value
            if value > high:
                value = 2 * high - value
        return value

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
        from itertools import product as cartesian_product

        grid_spec = getattr(self, 'grid', {})
        if not grid_spec:
            raise ValueError("prior='grid' requires a 'grid' section in the YAML config")

        # Build 1-D arrays for each grid parameter
        grid_axes = {}
        for param, values in grid_spec.items():
            grid_axes[param] = np.array(values, dtype=float)
            log.info(f"Grid axis: {param} → {len(values)} points")

        # Cartesian product of all grid axes
        axis_names = list(grid_axes.keys())
        axis_arrays = [grid_axes[n] for n in axis_names]
        grid_points = list(cartesian_product(*axis_arrays))
        n_grid = len(grid_points)
        log.info(f"Grid total: {n_grid} points from {len(axis_names)} axes")

        if n_grid > 10000:
            log.fatal(f"Grid produces {n_grid} points — exceeds the 10,000 model limit! "
                       "Reduce grid axes or split into multiple jobs using genModels_condor.py.")
            exit()

        # Override num_models
        self.num_models = n_grid

        # Populate self.points
        for i, name in enumerate(axis_names):
            self.points[name] = np.array([pt[i] for pt in grid_points])

        # Random-sample remaining parameters from their ranges
        for key, scanrange in self.parameters.items():
            if key in grid_spec:
                continue
            if isinstance(scanrange[0], list):
                # Interrupted range — weighted sub-ranges
                widths = [sr[1] - sr[0] for sr in scanrange]
                total = sum(widths)
                probs = [w / total for w in widths]
                vals = np.empty(n_grid)
                for j in range(n_grid):
                    idx = np.random.choice(len(scanrange), p=probs)
                    vals[j] = np.random.uniform(scanrange[idx][0], scanrange[idx][1])
                self.points[key] = vals
            elif scanrange[0] == scanrange[1]:
                # Fixed value
                self.points[key] = np.full(n_grid, scanrange[0])
            else:
                self.points[key] = np.random.uniform(scanrange[0], scanrange[1], n_grid)

        log.info(f"Grid sampling complete: {n_grid} models")

    # ── End of grid sampling ──────────────────────────────────────────

    def run_step(self, modelnum: int, **kwargs) -> bool:
        '''Function decorator for exectuting run_* function. Returns success bool.'''
        step_name = kwargs['name']
        
        # Select run_ function based on step name
        if hasattr(self, f"run_{step_name}") and callable(func := getattr(self, f"run_{step_name}")):
            return func(modelnum, **kwargs)
        else:
            raise KeyError(f"step {step_name} not supported!")
    
    def run_prep_input(self, modelnum: int, **kwargs) -> bool:
        '''Prep input. Returns the filename of the prepped file.'''
        
        rawfile = pyslha.read(self.rawfilen, ignorenomass = True)
        
        rawfile.blocks['EXTPAR'][1] = self.points['M_1'][modelnum]
        rawfile.blocks['EXTPAR'][2] = self.points['M_2'][modelnum]
        rawfile.blocks['EXTPAR'][3] = self.points['M_3'][modelnum]
        rawfile.blocks['EXTPAR'][11] = self.points['AT'][modelnum]
        rawfile.blocks['EXTPAR'][12] = self.points['Ab'][modelnum]
        rawfile.blocks['EXTPAR'][13] = self.points['Atau'][modelnum]
        rawfile.blocks['EXTPAR'][23] = self.points['mu'][modelnum]
        rawfile.blocks['EXTPAR'][25] = self.points['tanb'][modelnum]
        rawfile.blocks['EXTPAR'][26] = self.points['mA'][modelnum]
        rawfile.blocks['EXTPAR'][31] = self.points['meL'][modelnum]
        rawfile.blocks['EXTPAR'][32] = self.points['meL'][modelnum] # mmuL := meL
        rawfile.blocks['EXTPAR'][33] = self.points['mtauL'][modelnum]
        rawfile.blocks['EXTPAR'][34] = self.points['meR'][modelnum]
        rawfile.blocks['EXTPAR'][35] = self.points['meR'][modelnum] # mmuR := meR
        rawfile.blocks['EXTPAR'][36] = self.points['mtauR'][modelnum]
        rawfile.blocks['EXTPAR'][41] = self.points['mqL1'][modelnum]
        rawfile.blocks['EXTPAR'][42] = self.points['mqL1'][modelnum] # mqL2 := mqL1
        rawfile.blocks['EXTPAR'][43] = self.points['mqL3'][modelnum]
        rawfile.blocks['EXTPAR'][44] = self.points['muR'][modelnum]
        rawfile.blocks['EXTPAR'][45] = self.points['muR'][modelnum] # mcR := muR
        rawfile.blocks['EXTPAR'][46] = self.points['mtR'][modelnum]
        rawfile.blocks['EXTPAR'][47] = self.points['mdR'][modelnum]
        rawfile.blocks['EXTPAR'][48] = self.points['mdR'][modelnum] # msR := mdR
        rawfile.blocks['EXTPAR'][49] = self.points['mbR'][modelnum]
        
        preppedfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha" 
        pyslha.write(preppedfile, rawfile)
        
        return True
    
    def run_SPheno(self, modelnum: int, **kwargs) -> bool:
        '''Run SPheno and return True if successful. Re-add input blocks to output file if successful. Relevant for running tools on output file.'''
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        logfile = f"{self.scan_dir}/{kwargs['log_dir']}/{modelnum}.log"
        success = False
        
        # run SPheno
        os.system(f"SPheno {infile} {outfile} &> {logfile}")

        # Check that output file was actually produced
        if not os.path.exists(outfile):
            success = False
        else:
            # Check logfile for known error messages
            with open(logfile, 'r') as file: loglines = file.read()
            if "There has been a problem during the run." in loglines.strip() or "The error has occured" in loglines.strip(): success = False
            else: success = True

        # Validate output using same pyslha parsing as ntupling readSLHA
        if success:
            try:
                pyslha.readSLHAFile(outfile, ignorenomass=True)
            except:
                success = False

        # Re-add input blocks needed for other tools since SPheno swallows them:
        if success: addinputblocks(infile=outfile, blocksfile=self.rawfilen)
        
        return success
    
    def run_softsusy(self, modelnum: int, **kwargs) -> bool:
        '''Run softsusy and return True if successful.'''
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        success = False
        
        retcode = os.system(f"softpoint.x leshouches < {infile} &> {outfile}")
        if retcode == 0: success = True

        # Validate output using same pyslha parsing as ntupling readSLHA
        if success:
            try:
                pyslha.readSLHAFile(outfile, ignorenomass=True)
            except:
                success = False

        return success
    
    def run_micromegas(self, modelnum: int, **kwargs) -> bool:
        '''Run micromegas and return True if successful.'''
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile_raw = f"{self.scan_dir}/{kwargs['output_dir']}_raw/{modelnum}.out"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.csv"
        
        # Run micromegas to get the raw, humanly readable output
        os.system(f"main {infile} > {outfile_raw}")
        
        # Extract values into outfile
        success = microextract(infilen = outfile_raw, outfilen = outfile)
        
        return success
    
    def run_superiso(self, modelnum: int, **kwargs) -> bool:
        '''Run superiso and return True if successful.'''
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.flha"
        success = False
        
        # Run superiso and pipe humanly readable output into null
        os.system(f"slha.x {infile} &>/dev/null")
        # Move the auto-generated output file into the right location
        retcode = os.system(f"mv output.flha {outfile} &>/dev/null")
        if retcode == 0: success = True
        
        return success
    
    # def run_feynhiggs(self, modelnum: int, **kwargs) -> bool:
    #     '''Run FeynHiggs and return True if successful.'''

    #     infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
    #     outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
    #     logfile = f"{self.scan_dir}/{kwargs['log_dir']}/{modelnum}.log"
    #     success = False

    #     # Run FeynHiggs - it writes output to stdout which we capture in outfile
    #     retcode = os.system(f"FeynHiggs {infile} > {outfile} 2> {logfile}")
    #     if retcode == 0: success = True

    #     return success
    def run_feynhiggs(self, modelnum: int, **kwargs) -> bool:
        '''Run FeynHiggs and return True if successful. Re-add input blocks to output file if successful.'''

        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        logfile = f"{self.scan_dir}/{kwargs['log_dir']}/{modelnum}.log"
        success = False

        # Run FeynHiggs with default flags (42423120 = full MSSM, 2-loop, recommended settings)
        # FeynHiggs creates output file with .fh-001 extension automatically
        retcode = os.system(f"FeynHiggs {infile} 42423120 &> {logfile}")

        # Check return code first (simple success indicator)
        if retcode != 0:
            return False

        # FeynHiggs writes output to inputfile.fh-001
        fh_output = f"{infile}.fh-001"
        if not os.path.exists(fh_output):
            return False

        # Parse logfile for error indicators
        with open(logfile, 'r') as file:
            loglines = file.read()

        # Check for common FeynHiggs error patterns
        error_patterns = [
            "ERROR",
            "FAILED",
            "Segmentation fault",
            "invalid",
            "cannot",
            "Error on line"
        ]

        if any(pattern in loglines for pattern in error_patterns):
            success = False
        else:
            success = True

        # Move FeynHiggs output to the expected location
        if success:
            os.system(f"mv {fh_output} {outfile}")
            # Re-add input blocks that FeynHiggs might remove (like SPheno does)
            addinputblocks(infile=outfile, blocksfile=self.rawfilen)

        return success
    
    def run_gm2calc(self, modelnum: int, **kwargs) -> bool:
        '''Run GM2Calc and return True if successful.'''
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.slha"
        success = False
        
        # Run superiso and pipe humanly readable output into null
        retcode = os.system(f"gm2calc.x --slha-input-file={infile} > {outfile}")
        if retcode == 0: success = True
        
        return success
    
    def run_evade(self, modelnum: int, **kwargs) -> bool:
        '''Run EVADE and return True if successful.'''
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outfile = f"{self.scan_dir}/{kwargs['output_dir']}/{modelnum}.tsv"
        evade_conffile = f"{self.scan_dir}/{kwargs['output_dir']}_config/{modelnum}.cfg"
        evade_infile = f"{self.scan_dir}/{kwargs['output_dir']}_evade_in/{modelnum}.in"
        success = False
        
        # Convert .slha file into EVADE compatible input
        convert_slha(infile, evade_infile, evade_conffile, outfile)
        
        # Run evade and pipe humanly readable output into null
        retcode = os.system(f"EVADE_MSSM {evade_conffile} &>/dev/null")
        if retcode == 0: success = True
        
        return success
    
    def run_SModelS(self, modelnum: int, **kwargs) -> bool:
        '''Run SModelS and return True if successful.'''
        
        # Replace handlers of the library
        lib_logger = logging.getLogger("smodels")
        lib_logger.handlers = [StructlogHandler()]
        lib_logger.propagate = False
        lib_logger.setLevel(logging.INFO)
        
        infile = f"{self.scan_dir}/{kwargs['input_dir']}/{modelnum}.slha"
        outputDir = f"{self.scan_dir}/{kwargs['output_dir']}/"
        smodels_config = f"{datadir}/SModelS_parameters.ini"
        success = False
        
        # Import those parts of smodels that are needed for this exercise
        from smodels.matching import modelTester
        from smodels.base import runtime
        from smodels.tools import particlesLoader
        from importlib import reload
        
        # Import those parts of smodels that are needed for this exercise
        from smodels.tools import xsecComputer
        from smodels.base.physicsUnits import TeV, fb
        from smodels.installation import installDirectory
        from smodels.tools.xsecComputer import LO, NLL
        # logging.getLogger("smodels").setLevel(logging.CRITICAL) # TODO: Need to find a way to clean up SModelS printout
        
        # Lets compute the NLL cross-sections for 13 TeV. The xsecComputer will first use Pythia to compute
        # the LO cross-sections and then NLLfast to compute the k-factors.
        # The output will contain only the processes contained in NLLfast (gluino and squark production)
        # For the Pythia step we have to define the number of MC events (1k)
        # (To see how to also include LO cross-sections for all processes check the LO HowTo)
        computer = xsecComputer.XSecComputer ( maxOrder=NLL, nevents = 10_000, pythiaVersion=8 )
        
        # Try computing xsecsNLL, will fail for missing inputs
        try:
            xsecsNLL=computer.compute(sqrts = 13*TeV, slhafile=infile )
        except:
            success = False
            return success
        
        # If all relevant xsecs are 0, computer.compute() will return an empty list. Set SModelS to fail in this case:
        if xsecsNLL == []:
            success = False
            return success
        
        # Finally, lets write the cross-sections back to the file 
        # (will write only the cross-sections not overlapping the existing ones):
        computer.addXSecToFile(xsecsNLL,infile)
        
        #Read and check parameter file, exit parameterFile does not exist
        parser = modelTester.getParameters(smodels_config)
        
        #Change option:
        # parser.set('printer','outputtype','summary')
        parser.set('printer','outputtype','python')
        
        #Determine particles module from ini file, if necessary """
        if parser.has_option("particles","model"):
            runtime.modelFile = parser.get( "particles", "model" )
            reload(particlesLoader)

        #Load the database defined in the parameters file, exit if not found """
        database = modelTester.loadDatabase(parser,db=None)
        
        #run SModelS with input file:
        output = modelTester.testPoint(infile, outputDir, parser, database)
        
        # As we're not exiting the program, let's make sure all output is flushed
        [ x.flush() for x in output.values() ]
        
        success = True
        return success
    
    def generate_models(self) -> None:
        '''Main function to generate models.'''
        
        log.info(f"Starting Model Generation...")
        
        # Set up directories for saving scan
        log.info(f"Setting up output directories for scan")
        os.system(f"rm -r {self.scan_dir} &>/dev/null")
        os.makedirs(self.scan_dir)
        
        for step in self.steps:
            dirns = ['output_dir', 'log_dir']
            for dirn in dirns:
                if dirn in step: os.makedirs(f"{self.scan_dir}/{step[dirn]}")
                if dirn in step and "micromegas" in step['name']: os.makedirs(f"{self.scan_dir}/{step[dirn]}_raw")
                if dirn in step and "evade" in step['name']: 
                    os.makedirs(f"{self.scan_dir}/{step[dirn]}_config")
                    os.makedirs(f"{self.scan_dir}/{step[dirn]}_evade_in")
        
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
        else:
            raise ValueError(f"ERROR: prior {self.prior} not supported. Can be: flat, fixed, mcmc, grid.")

        # prep dict to keep track of successes:
        succcounts = {}
        for step in self.steps:
            succcounts[step['name']] = 0

        for mod in range(self.num_models):
            log.info(f"Generating Model: {mod}.slha ({mod+1}/{self.num_models})")

            model_ok = True
            for step in self.steps:
                if not model_ok:
                    log.info(f"\tskipping step: {step['name']} (previous step failed)")
                    continue
                log.info(f"\trunning step: {step['name']}")
                success = self.run_step(modelnum=mod, **step)

                # Keep track of successes
                if success: succcounts[step['name']] += 1
                else: model_ok = False

        self._dump_scan_config()

        log.info("Finished generating models! Stats for successful generations (successful/attempted):")
        for step in succcounts:
            log.info(f"\t{step}: {succcounts[step]}/{self.num_models}.")

        return None

    def _dump_scan_config(self) -> None:
        '''Dump scan configuration to YAML file in scan directory.'''
        with open(f"{self.scan_dir}/scan_config.yaml", "w") as file:
            skipkeys = ['points']
            dumpdict = {key: value for key, value in vars(self).items() if key not in skipkeys}
            yaml.dump(dumpdict, file, default_flow_style=None, encoding=None)
    
    def mkntuple(self) -> None:
        '''Promote ntuple making to class attribute.'''
        
        return mkntuple(self.steps, self.scan_dir, self.num_models, self.isGMSB)