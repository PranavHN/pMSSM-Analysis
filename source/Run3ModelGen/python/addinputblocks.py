import pyslha
import fileinput
import numpy as np

def addinputblocks(infile: str, blocksfile: str) -> None:
    '''Function for adding input blocks defined in blocksfile to infile. Needed since SPheno scraps unrecognised blocks from files while processing.'''

    filedict = pyslha.read(infile, ignorenomass=True, ignoreblocks=['SPINFO'])
    blocksdict = pyslha.read(blocksfile, ignorenomass=True)

    blocksdict.blocks["SOFTSUSY"][28] = filedict.blocks["MASS"][25] #mh(pole)
    blocksdict.blocks["SOFTSUSY"][29] = filedict.blocks["MASS"][36] #mA0(pole)
    blocksdict.blocks["SOFTSUSY"][30] = filedict.blocks["MASS"][35] #mH0(pole)
    blocksdict.blocks["SOFTSUSY"][31] = filedict.blocks["MASS"][37] #mH+/-(pole)

    blocksdict.blocks["SPHENOINPUT"][3] = 1 # force SPheno to use external mass spectrum
    scale = np.sqrt(filedict.blocks["MASS"][1000006]*filedict.blocks["MASS"][2000006]) # Q = sqrt(m_t_1*m_t_2)

    # Add blocks to infile
    blocklist = ["GM2CALCCONFIG", "SOFTSUSY", "SPHENOINPUT"]
    for blk in blocklist:
        filedict.blocks[blk] = blocksdict.blocks[blk]
        
    # Override SUSY scale in EXTPAR 0 for SPheno (scale not recalculated if external mass spectrum given)
    filedict.blocks["EXTPAR"][0] = scale
    
    # Add gravitino mass if included in blocksdict
    if 1000039 in blocksdict.blocks["EXTPAR"]: 
        filedict.blocks["EXTPAR"][1000039] = blocksdict.blocks["EXTPAR"][1000039]
        
    # Override infile
    pyslha.write(infile, filedict)