#!/usr/bin/env python

import click
from Run3ModelGen.condor_handler import CondorHandler

@click.command()
@click.option('--config_file', default=None, help='Yaml file containing scan configuration. If no config is provided, will read from the default.')
@click.option('--condor_dir', default="genModels_condor_log", type=str, help='Log dir containing condor info.')
@click.option('--num_jobs', default=2, type=int, help='Number of batch jobs to submit.')
@click.option('--flavour', default="tomorrow", type=str, help='JobFlavour for condor jobs. Use "espresso" for testing. Defaults to "tomorrow')
@click.option('--force', is_flag=True, type=bool, help='Force overriding condor_dir.')
def main(config_file, condor_dir, num_jobs, flavour, force):
    '''Main script for submitting condor jobs for model generation.'''

    # Initialise CondorHandler for handling submission
    handler = CondorHandler(mode="genModels", config_file=config_file, condor_dir=condor_dir, num_jobs=num_jobs, flavour=flavour, force=force)
    
    # Generate submission files
    handler.prep_subfile()
    
    # Submit condor jobs
    handler.submit_jobs()

if __name__ == "__main__":
    main()
