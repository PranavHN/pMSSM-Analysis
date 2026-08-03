'''Module for extracting micromegas values from humanly readable output into csv file'''

import os
import re

# micrOMEGAs determines the lightest odd particle generically (sortOddParticles) and
# computes the freeze-out abundance of WHATEVER it is -- charge included; the stock
# charged-DM abort in MSSM/main.c is commented out. It reports the identity as
#     Dark matter candidate is '~o1' with spin=1/2  mass=1.23E+02
# and additionally prints " ~o1 is not CDM" when the LSP is not the lightest
# neutralino. Neither was captured, so relic-density scans silently accepted
# sneutrino- and slepton-LSP spectra whose Omega satisfied the constraint. Emit the
# identity as a NUMERIC flag so it can be used as an ordinary constraint observable
# (1 = neutralino LSP, 0 = anything else); absent line -> no key -> the point is
# treated as unevaluable, i.e. fail closed.
_CDM_RE = re.compile(r"Dark matter candidate is\s*'([^']+)'")
_CDM_MASS_RE = re.compile(r"mass\s*=\s*([0-9.eE+-]+)")
NEUTRALINO_CDM = "~o1"

def microextract(infilen: str, outfilen: str) -> bool:
    '''Function for extracting micromegas values from infilen into outfilen. Returns success code'''
    
    varlist = ["deltartho","gmuon","bsgnlo","bsgnlo_SM","bsmumu","btaunu","bmunu","Rl23","Xf","Omega","Omega2"]
    correctline = False
    foundcrs = False

    file = open(infilen)
    newfile = open(outfilen, 'w')
    
    foundvars = 0
    newfile.write("Variable,MO\n")
    for line in file.readlines():
        for i in range(len(varlist)):
            try:
                line.split().index(varlist[i])
                correctline = True
                foundvars+=1
            except:
                line
        
            if correctline:
                if varlist[i]=='deltartho':
                    newfile.write('deltarho,'+str(line.split()[line.split().index(varlist[i])+1])+'\n')
                else:
                    newfile.write(varlist[i]+','+str(line.split()[line.split().index(varlist[i])+1])+'\n')
                correctline = False
        # LSP identity (see _CDM_RE above). Deliberately not counted in foundvars:
        # it must not by itself make an otherwise empty extraction look successful.
        cdm = _CDM_RE.search(line)
        if cdm:
            newfile.write(f"cdm_is_neutralino,{1 if cdm.group(1) == NEUTRALINO_CDM else 0}\n")
            mass = _CDM_MASS_RE.search(line)
            if mass:
                newfile.write(f"cdm_mass,{mass.group(1)}\n")

        if "[pb]" in line:
            foundcrs = True
        if 'proton' in line and foundcrs:
            newfile.write('proton_SI,'+line.split()[2]+'\n')
            #print line.split()[2]
            newfile.write('proton_SD,'+line.split()[4]+'\n')
        if 'neutron' in line and foundcrs:
            newfile.write('neutron_SI,'+line.split()[2]+'\n')
            newfile.write('neutron_SD,'+line.split()[4]+'\n')
            foundcrs = False

    if foundvars == 0:
        #newfile.write("ERROR: No values have been found!")
        newfile.close()
        os.remove(newfile.name)
        
        return False
    
    else: return True
