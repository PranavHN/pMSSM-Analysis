import pyslha
import fileinput
import numpy as np


class UnusableSpectrum(ValueError):
    '''A spectrum whose masses cannot be used to build the input blocks.

    Raised instead of crashing: pathological points do occur (a few per 10,000),
    and the tool wrappers turn this into an ordinary per-model failure.
    '''


def _finite_mass(filedict, pdg: int, infile: str) -> float:
    '''Read one MASS entry as a finite float.

    The generators write "NaN" (and Fortran can write "****" on field overflow)
    for pathological points; pyslha then hands the entry back as a string, which
    used to blow up the arithmetic below — and with it the whole batch.
    '''
    try:
        value = float(filedict.blocks["MASS"][pdg])
    except (KeyError, TypeError, ValueError) as exc:
        entry = filedict.blocks["MASS"].entries.get(pdg, "<missing>") if "MASS" in filedict.blocks else "<no MASS block>"
        raise UnusableSpectrum(f"MASS entry {pdg} in {infile} is not a number ({entry!r})") from exc
    if not np.isfinite(value):
        raise UnusableSpectrum(f"MASS entry {pdg} in {infile} is {value}")
    return value


def addinputblocks(infile: str, blocksfile: str) -> None:
    '''Function for adding input blocks defined in blocksfile to infile. Needed since SPheno scraps unrecognised blocks from files while processing.'''

    filedict = pyslha.read(infile, ignorenomass=True, ignoreblocks=['SPINFO'])
    blocksdict = pyslha.read(blocksfile, ignorenomass=True)

    # Validate before touching the file: everything below rewrites infile in
    # place, so a spectrum we cannot use must be rejected while it is still intact.
    m_stop1 = _finite_mass(filedict, 1000006, infile)
    m_stop2 = _finite_mass(filedict, 2000006, infile)

    blocksdict.blocks["SOFTSUSY"][28] = filedict.blocks["MASS"][25] #mh(pole)
    blocksdict.blocks["SOFTSUSY"][29] = filedict.blocks["MASS"][36] #mA0(pole)
    blocksdict.blocks["SOFTSUSY"][30] = filedict.blocks["MASS"][35] #mH0(pole)
    blocksdict.blocks["SOFTSUSY"][31] = filedict.blocks["MASS"][37] #mH+/-(pole)

    blocksdict.blocks["SPHENOINPUT"][3] = 1 # force SPheno to use external mass spectrum
    scale = np.sqrt(m_stop1*m_stop2) # Q = sqrt(m_t_1*m_t_2)

    blocklist = ["GM2CALCCONFIG", "SOFTSUSY", "SPHENOINPUT"]
    f = fileinput.input(infile, inplace=1)
    foundextpar = False
    for xline in f:
        xline = xline.strip('\n')
        
        if f.isfirstline():
            #include input blocks in beginning of file:
            for blk in blocklist:
                print("BLOCK {}".format(blk))
                for key in blocksdict.blocks[blk].keys():
                    print("\t{}\t{}".format(key, blocksdict.blocks[blk][key]))
            print(xline)
        else:
            if "EXTPAR" in xline:
                foundextpar = True
            if foundextpar and "0" in xline:
                # Add gravitino mass if included in blocksdict
                if 1000039 in blocksdict.blocks["EXTPAR"]:
                    print("\t{}\t{}\t# gravitino".format(1000039, blocksdict.blocks["EXTPAR"][1000039]))
                # overwrite SUSY scale in EXTPAR 0 for SPheno (scale not recalculated if external mass spectrum given)
                print("\t{}\t{}\t# Q".format(0, scale))
                foundextpar = False
            else:
                print(xline)