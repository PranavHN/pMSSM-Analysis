#!/usr/bin/env python

import os
import shutil
import click

@click.command()
@click.argument('prior', type=click.Choice(['flat', 'fixed', 'mcmc'], case_sensitive=False))
@click.option('--output_dir', default='.', type=str, help='Directory to save the config file. Defaults to current working directory.')
def main(prior, output_dir):
    '''Generate an example config file for the given prior type (flat, fixed, or mcmc).'''

    datadir = os.environ['DATAPATH'].split(':')[0]
    config_map = {
        'flat':  'example_flat_config.yaml',
        'fixed': 'example_fixed_config.yaml',
        'mcmc':  'example_mcmc_config.yaml',
    }

    src = os.path.join(datadir, 'configs', config_map[prior])
    dst = os.path.join(output_dir, config_map[prior])

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(src, dst)
    click.echo(f"Saved example {prior} config to {dst}")

if __name__ == "__main__":
    main()
