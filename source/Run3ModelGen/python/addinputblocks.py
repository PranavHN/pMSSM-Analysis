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