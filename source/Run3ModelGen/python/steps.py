'''Per-tool pipeline step functions on explicit paths.

Each function runs ONE external tool for ONE model and returns a success bool.
The bodies are lifted verbatim from ModelGenerator.run_* (modelgen.py) so the
legacy in-process path and the Snakemake per-batch rules execute the SAME code —
ModelGenerator.run_* are now one-line delegates into this module.

Per-model tool failure is EXPECTED for random pMSSM points and is data, not an
error: callers record the bool and skip downstream steps for that model.

No environment variables are read at import time; all paths are arguments.
'''

import os
import shutil
import subprocess
import tempfile

import pyslha

from Run3ModelGen.microextract import microextract
from Run3ModelGen.addinputblocks import addinputblocks, UnusableSpectrum
from Run3ModelGen.pMSSM_convert import convert_slha

import structlog
import logging
log = structlog.get_logger()


# Bridge handler: this forwards standard log records → structlog
class StructlogHandler(logging.Handler):
    def emit(self, record):
        # Convert logging.LogRecord → structlog event
        log.bind(module=record.name).info(record.getMessage())


def run_prep_input(point: dict, rawfilen: str, outfile: str) -> bool:
    '''Prep input: fill the raw SLHA template's EXTPAR with one sampled point.

    :param point: {parameter name: value} for one model (1st/2nd-gen sfermions tied below).
    '''

    rawfile = pyslha.read(rawfilen, ignorenomass = True)

    rawfile.blocks['EXTPAR'][1] = point['M_1']
    rawfile.blocks['EXTPAR'][2] = point['M_2']
    rawfile.blocks['EXTPAR'][3] = point['M_3']
    rawfile.blocks['EXTPAR'][11] = point['AT']
    rawfile.blocks['EXTPAR'][12] = point['Ab']
    rawfile.blocks['EXTPAR'][13] = point['Atau']
    rawfile.blocks['EXTPAR'][23] = point['mu']
    rawfile.blocks['EXTPAR'][25] = point['tanb']
    rawfile.blocks['EXTPAR'][26] = point['mA']
    rawfile.blocks['EXTPAR'][31] = point['meL']
    rawfile.blocks['EXTPAR'][32] = point['meL'] # mmuL := meL
    rawfile.blocks['EXTPAR'][33] = point['mtauL']
    rawfile.blocks['EXTPAR'][34] = point['meR']
    rawfile.blocks['EXTPAR'][35] = point['meR'] # mmuR := meR
    rawfile.blocks['EXTPAR'][36] = point['mtauR']
    rawfile.blocks['EXTPAR'][41] = point['mqL1']
    rawfile.blocks['EXTPAR'][42] = point['mqL1'] # mqL2 := mqL1
    rawfile.blocks['EXTPAR'][43] = point['mqL3']
    rawfile.blocks['EXTPAR'][44] = point['muR']
    rawfile.blocks['EXTPAR'][45] = point['muR'] # mcR := muR
    rawfile.blocks['EXTPAR'][46] = point['mtR']
    rawfile.blocks['EXTPAR'][47] = point['mdR']
    rawfile.blocks['EXTPAR'][48] = point['mdR'] # msR := mdR
    rawfile.blocks['EXTPAR'][49] = point['mbR']

    pyslha.write(outfile, rawfile)

    return True


def run_SPheno(infile: str, outfile: str, logfile: str, rawfilen: str) -> bool:
    '''Run SPheno and return True if successful. Re-add input blocks to output file if successful. Relevant for running tools on output file.

    SPheno reads its filename arguments into fixed-length Fortran buffers that
    hold 120 characters; a longer path is truncated, so it silently computes
    garbage and fails with "matrix contains NaN" — indistinguishable from an
    unphysical parameter point. It is therefore run inside a private working
    directory with SHORT RELATIVE filenames, which keeps it immune to however
    deep the scan directory happens to live (an EOS path, a dated run
    directory, ...). The workdir also contains the scratch files SPheno drops
    into its cwd, so concurrent models cannot collide.
    '''

    success = False

    workdir = tempfile.mkdtemp(prefix="spheno_")
    try:
        local_in, local_out = "in.slha", "out.slha"
        shutil.copy2(infile, os.path.join(workdir, local_in))

        # run SPheno
        # NB: not named `log` — that would shadow the module-level logger for
        # the whole function (Python scopes assigned names function-wide).
        with open(logfile, 'wb') as logfh:
            subprocess.run(["SPheno", local_in, local_out], cwd=workdir,
                           stdout=logfh, stderr=subprocess.STDOUT)

        # Hand the result over whether or not the run succeeded: previously SPheno
        # wrote straight into the scan dir, so a failed model left its (partial)
        # output there, and both the ntuple and the published tarball reflect
        # that. Everything below is then identical to the pre-workdir version.
        produced = os.path.join(workdir, local_out)
        if os.path.exists(produced):
            shutil.move(produced, outfile)

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

        # Re-add input blocks needed for other tools since SPheno swallows them.
        # A spectrum with unusable masses (NaN and friends) is a failed model,
        # not a failed run — a few such points per 10,000 are normal.
        if success:
            try:
                addinputblocks(infile=outfile, blocksfile=rawfilen)
            except UnusableSpectrum as exc:
                log.warning(f"SPheno output rejected: {exc}")
                success = False

        return success
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_softsusy(infile: str, outfile: str) -> bool:
    '''Run softsusy and return True if successful.'''

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


def run_micromegas(infile: str, outfile_raw: str, outfile: str) -> bool:
    '''Run micromegas and return True if successful.'''

    # Run micromegas to get the raw, humanly readable output
    os.system(f"main {infile} > {outfile_raw}")

    # Extract values into outfile
    success = microextract(infilen = outfile_raw, outfilen = outfile)

    return success


def run_superiso(infile: str, outfile: str) -> bool:
    '''Run superiso and return True if successful.

    slha.x always writes its result to the FIXED name ``output.flha`` in the
    current working directory — under the legacy shared-cwd loop two concurrent
    scans would clobber each other. Run it in a private temporary workdir
    instead; the produced file is byte-identical.
    '''

    workdir = tempfile.mkdtemp(prefix="superiso_")
    try:
        # Run superiso and pipe humanly readable output into null
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(["slha.x", os.path.abspath(infile)], cwd=workdir,
                           stdout=devnull, stderr=devnull)
        # Move the auto-generated output file into the right location
        produced = os.path.join(workdir, "output.flha")
        if not os.path.exists(produced):
            return False
        shutil.move(produced, outfile)
        return True
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_feynhiggs(infile: str, outfile: str, logfile: str, rawfilen: str) -> bool:
    '''Run FeynHiggs and return True if successful. Re-add input blocks to output file if successful.

    FeynHiggs writes its result to ``<input>.fh-001`` NEXT TO the input file —
    run it on a private copy in a temporary workdir so the side-file never
    lands in (or, on failure, litters) the upstream SPheno/ directory.
    '''

    success = False

    workdir = tempfile.mkdtemp(prefix="feynhiggs_")
    try:
        local_in = os.path.join(workdir, os.path.basename(infile))
        shutil.copy2(infile, local_in)

        # Run FeynHiggs with default flags (42423120 = full MSSM, 2-loop, recommended settings)
        # FeynHiggs creates output file with .fh-001 extension automatically
        retcode = os.system(f"FeynHiggs {local_in} 42423120 &> {logfile}")

        # Check return code first (simple success indicator)
        if retcode != 0:
            return False

        # FeynHiggs writes output to inputfile.fh-001
        fh_output = f"{local_in}.fh-001"
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
            shutil.move(fh_output, outfile)
            # Re-add input blocks that FeynHiggs might remove (like SPheno does).
            # DELIBERATE, even though nothing in the pipeline currently reads this
            # file (no step declares feynhiggs as its input_dir, and the ntuple
            # takes only MASS[25]/DMASS[25] from it): keeping the output a
            # self-contained SLHA input is what allows further calculations to be
            # run on top of FeynHiggs later, the same way gm2calc et al. run on
            # top of SPheno. Do not "simplify" it away.
            # FeynHiggs writes NaN masses for pathological points; that is a
            # failed model, not a failed run (it used to kill the whole batch).
            try:
                addinputblocks(infile=outfile, blocksfile=rawfilen)
            except UnusableSpectrum as exc:
                log.warning(f"FeynHiggs output rejected: {exc}")
                success = False

        return success
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_gm2calc(infile: str, outfile: str) -> bool:
    '''Run GM2Calc and return True if successful.'''

    success = False

    # Run superiso and pipe humanly readable output into null
    retcode = os.system(f"gm2calc.x --slha-input-file={infile} > {outfile}")
    if retcode == 0: success = True

    return success


def run_evade(infile: str, outfile: str, evade_conffile: str, evade_infile: str) -> bool:
    '''Run EVADE and return True if successful.'''

    success = False

    # Convert .slha file into EVADE compatible input
    convert_slha(infile, evade_infile, evade_conffile, outfile)

    # Run evade and pipe humanly readable output into null
    retcode = os.system(f"EVADE_MSSM {evade_conffile} &>/dev/null")
    if retcode == 0: success = True

    return success


# One SModelS parameter set + results database per PROCESS, not per model.
# loadDatabase unpickles ~1.4 GB and the memory is not returned afterwards, so
# reloading it for every model grew RSS by ~170 MB per model and got a
# 10,000-model batch OOM-killed after 46 of them (7.9 GB of an 8 GB request).
# smodels' own batch entrypoint (modelTester.testPoints) takes an
# already-loaded database for exactly this reason; per-model loading also made
# each model pay the unpickling cost, which dominated the per-model runtime.
# Keyed by ini path: a different parameter file means a different session.
_SMODELS_SESSIONS = {}


def _smodels_session(smodels_ini: str):
    '''Return the (parser, database) pair for this parameter file, loading once.'''
    session = _SMODELS_SESSIONS.get(smodels_ini)
    if session is None:
        from smodels.matching import modelTester
        from smodels.base import runtime
        from smodels.tools import particlesLoader
        from importlib import reload

        #Read and check parameter file, exit parameterFile does not exist
        parser = modelTester.getParameters(smodels_ini)

        #Change option:
        # parser.set('printer','outputtype','summary')
        parser.set('printer','outputtype','python')

        #Determine particles module from ini file, if necessary """
        if parser.has_option("particles","model"):
            runtime.modelFile = parser.get( "particles", "model" )
            reload(particlesLoader)

        #Load the database defined in the parameters file, exit if not found """
        database = modelTester.loadDatabase(parser,db=None)

        session = (parser, database)
        _SMODELS_SESSIONS[smodels_ini] = session
    return session


def run_SModelS(infile: str, output_dir: str, smodels_ini: str, xsec_dest: str = None) -> bool:
    '''Run SModelS and return True if successful.

    :param xsec_dest: where the computed NLL cross sections are written and
        SModelS is evaluated. ``None`` (legacy) appends them to ``infile`` IN
        PLACE — today's behavior in the sequential per-model loop, where every
        other SPheno consumer has already read the file. The Snakemake path
        passes a copy path instead, keeping the SPheno output immutable; the
        EOS packaging step folds the augmented copy back over the SPheno/ path
        inside the tarball so published tarballs are unchanged.
    '''

    # Replace handlers of the library
    lib_logger = logging.getLogger("smodels")
    lib_logger.handlers = [StructlogHandler()]
    lib_logger.propagate = False
    lib_logger.setLevel(logging.INFO)

    success = False

    # Import those parts of smodels that are needed for this exercise
    from smodels.matching import modelTester

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

    # The file that receives the cross sections and is passed to SModelS: the
    # SPheno output itself (legacy), or an immutable-input-preserving copy.
    if xsec_dest is None:
        xsec_dest = infile
    else:
        shutil.copy2(infile, xsec_dest)

    # Finally, lets write the cross-sections back to the file
    # (will write only the cross-sections not overlapping the existing ones):
    computer.addXSecToFile(xsecsNLL,xsec_dest)

    # Parameters + database, loaded once per process (see _smodels_session)
    parser, database = _smodels_session(smodels_ini)

    #run SModelS with input file:
    output = modelTester.testPoint(xsec_dest, output_dir, parser, database)

    # As we're not exiting the program, let's make sure all output is flushed
    [ x.flush() for x in output.values() ]

    success = True
    return success
