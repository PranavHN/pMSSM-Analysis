'''Module for handling batch jobs.'''

import os
datadir = os.environ['DATAPATH'].split(':')[0]

import structlog
import logging
log = structlog.get_logger()
structlog.stdlib.recreate_defaults(log_level=logging.INFO)  # so we have logger names

class CondorHandler:
    '''Class for batch job handling.'''
    def __init__(self, mode: str, **kwargs) -> None:
        
        # Print logo. Note: ASCII art generated with https://patorjk.com/software/taag/ (Small, Fitted)
        with open(f"{datadir}/logo.txt", 'r') as file:
            logo = "\n"+file.read()+"\n"
        print(logo)
        
        if mode not in ["genModels", "extractModels"]: 
            log.fatal(f"CondorHandler argument 'mode' {mode} not supported! Has to be either 'genModels' or 'extractModels'!")
            exit()
        
        self.mode = mode
        self.condor_dir = kwargs.get('condor_dir', f"{self.mode}_condor_log")
        self.force = kwargs.get('force', False)
        self.flavour = kwargs.get('flavour', 'tomorrow')
        
        if self.mode == "genModels":
            self.config_file = kwargs.get('config_file', None)
            if self.config_file is not None and not os.path.isfile(self.config_file):
                log.fatal(f"config file '{self.config_file}' does not exist or is not a file!")
                exit()
            self.num_jobs = kwargs.get('num_jobs', 2)

        elif self.mode == "extractModels":
            self.eos_dir = kwargs['eos_dir'] if not kwargs['eos_dir'].endswith('/') else kwargs['eos_dir'][:-1]
            self.ClusterId = self.eos_dir.split('/')[-1]
            self.selection = kwargs.get('selection', "akarr['SS_m_h']!=-1")
            self.num_jobs = kwargs.get('num_jobs', None)
            if self.num_jobs is None:
                self.num_jobs = len([subdir for subdir in os.listdir(self.eos_dir) if os.path.isdir(f"{self.eos_dir}/{subdir}")])
        
        # create log dir if it does not yet exist
        try:
            os.makedirs(f"{self.condor_dir}/outfiles")
        except:
            if not self.force:
                log.exception(f"condor_dir '{self.condor_dir}' already exists! Please either supply a different one or run with '--force' to override it.")
                exit()
            else:
                log.warning(f"condor_dir '{self.condor_dir}' already exists! Running in '--force' mode, so results may be overwritten.")
        
        log.info(f"Initialised CondorHandler with:")
        for var, varval in vars(self).items():
            log.info(f"\t{var} = {varval}")
        
        return None
    
    def prep_subfile(self) -> None:
        '''prepare submission files for condor run.'''
        
        if self.mode == "genModels":
        
            # Save config file in condor_dir
            if self.config_file is not None:
                os.system(f"cp {self.config_file} {self.condor_dir}/{self.config_file}")
                log.info(f"Saved config file in {self.condor_dir}/{self.config_file}.")
                confstr = f"{os.getcwd()}/{self.condor_dir}/{self.config_file}"
            else:
                os.system(f"cp {datadir}/configs/default_config.yaml {self.condor_dir}/")
                log.info(f"Saved config file in {self.condor_dir}/default_config.yaml.")
                confstr = "None"
            
            subfile_text = f"""executable = {self.condor_dir}/genmodels_condor.sh

Requirements = (Machine != "b6100c26e0.cern.ch")
arguments = $(ClusterId) $(ProcId) {os.environ['RUN3MODELGENDIR']} {confstr}
output = {self.condor_dir}/outfiles/genmodels.$(ClusterId).$(ProcId).out
error = {self.condor_dir}/outfiles/genmodels.$(ClusterId).$(ProcId).err
log = {self.condor_dir}/outfiles/genmodels.$(ClusterId).$(ProcId).log
transfer_output_files = ""
RequestCpus = 4
RequestDisk = 700000
+JobFlavour = "{self.flavour}"
+MaxRuntime = 604800
queue {self.num_jobs}"""

            # save subfile text in subfile
            with open(f"{self.condor_dir}/subcondor_genModels.sub", 'w') as subfile:
                subfile.write(subfile_text)
            
            log.info(f"Written condor submission file to {self.condor_dir}/subcondor_genModels.sub")
            
            # copy executeable to condor_dir
            os.system(f"cp {datadir}/genmodels_condor.sh {self.condor_dir}/genmodels_condor.sh")
            log.info(f"Written condor bash script to {self.condor_dir}/genmodels_condor.sh")
        
        elif self.mode == "extractModels":
            
            subfile_text = f"""executable = {self.condor_dir}/extractmodels_condor.sh

Requirements = (Machine != "b6100c26e0.cern.ch")
arguments = {self.ClusterId} $(ProcId) {os.environ['RUN3MODELGENDIR']} {self.eos_dir} {self.selection}
output = {self.condor_dir}/outfiles/extractmodels.{self.ClusterId}.$(ProcId).out
error = {self.condor_dir}/outfiles/extractmodels.{self.ClusterId}.$(ProcId).err
log = {self.condor_dir}/outfiles/extractmodels.{self.ClusterId}.$(ProcId).log
transfer_output_files = ""
RequestCpus = 4
RequestDisk = 700000
+JobFlavour = "{self.flavour}"
+MaxRuntime = 604800
queue {self.num_jobs}"""
    
            # save subfile text in subfile
            with open(f"{self.condor_dir}/subcondor_extractModels.sub", 'w') as subfile:
                subfile.write(subfile_text)
            
            log.info(f"Written condor submission file to {self.condor_dir}/subcondor_extractModels.sub")
            
            # copy executeable to condor_dir
            os.system(f"cp {datadir}/extractmodels_condor.sh {self.condor_dir}/extractmodels_condor.sh")
            log.info(f"Written condor bash script to {self.condor_dir}/extractmodels_condor.sh")
        
        return None
    
    def submit_jobs(self) -> None:
        '''submit prepared condor jobs.'''
        
        log.info(f"Submitting jobs to condor...")
        if self.mode == "genModels": subcode = os.system(f"condor_submit {self.condor_dir}/subcondor_genModels.sub > {self.condor_dir}/submission.log")
        elif self.mode == "extractModels": subcode = os.system(f"condor_submit {self.condor_dir}/subcondor_extractModels.sub > {self.condor_dir}/submission.log")
        
        with open(f"{self.condor_dir}/submission.log", 'r') as sublog:
            ClusterId = sublog.read().split()[-1].strip('.')
        
        if subcode == 0: 
            log.info(f"Submission to cluster {ClusterId} successful!")
            substr = f"0-{self.num_jobs-1}" if self.num_jobs > 1 else "0"
            
            if self.mode == "genModels": log.info(f"After your models have finished generating, the .tar ball and ntuple will be saved in {os.environ['EOSPATH']}/{ClusterId}/{substr}")
            elif self.mode == "extractModels": log.info(f"After your models have finished extracting, the Models directory to be copied into the pMSSMFactory will be saved in {os.environ['EOSPATH']}/SelectedModels/{self.ClusterId}/Models")
        else: log.error(f"Submission failed!")
        
        return None