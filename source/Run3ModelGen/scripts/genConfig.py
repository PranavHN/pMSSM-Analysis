#!/usr/bin/env python

import os
import shutil
import click

@click.command()
@click.argument('prior', type=click.Choice(['flat', 'fixed', 'mcmc', 'grid', 'importance', 'active_learning'], case_sensitive=False))
@click.option('--output_dir', default='.', type=str, help='Directory to save the config file. Defaults to current working directory.')
def main(prior, output_dir):
    '''Generate an example config file for the given prior type (flat, fixed, mcmc, grid, importance, or active_learning).'''

    datadir = os.environ['DATAPATH'].split(':')[0]
    config_map = {
        'flat':  'example_flat_config.yaml',
        'fixed': 'example_fixed_config.yaml',
        'mcmc':  'example_mcmc_config.yaml',
        'grid':  'example_grid_config.yaml',
        'importance': 'example_importance_config.yaml',
        'active_learning': 'example_active_learning_config.yaml',
    }

    src = os.path.join(datadir, 'configs', config_map[prior])
    dst = os.path.join(output_dir, config_map[prior])

    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(src, dst)
    click.echo(f"Saved example {prior} config to {dst}")

    # importance prior also needs the example seed CSV
    if prior == 'importance':
        seed_name = 'example_importance_seeds.csv'
        seed_src = os.path.join(datadir, 'configs', seed_name)
        seed_dst = os.path.join(output_dir, seed_name)
        shutil.copy2(seed_src, seed_dst)
        click.echo(f"Saved example seed file to {seed_dst}")

if __name__ == "__main__":
    main()
