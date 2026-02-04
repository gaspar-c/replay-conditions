"""
figS2C.py
Generates Figure S2, panel C.

p_rc and p_ff are varied from 0.00 to 0.17 in 18 steps each. 
Each (p_ff, p_rc) pair is run with multiple random connectivity seeds.
Thus, number of simulated networks is 18 x 18 x len(conn_seeds). 
Each network is stimulated num_stims times.

For quicker simulation times, this file uses as default conn_seeds = [1] and num_stims = 1.
In the manuscript, we used num_stims = 5 and conn_seeds = [1, 2, 3, 4, 5].

Run as 'python fig2C.py plot' to plot results from a previous run.
"""

import time
import os
import sys
import numpy as np
from spiking_networks.group_replay_analysis import get_correlations, group_plot, get_replay_pivot
from spiking_networks.model2 import run_simulation
from general_code.aux_functions import select_group_label_dir

ei_ratio = 8  # ratio of excitatory to inhibitory neurons

conn_seeds = [1]   # random seeds; in manuscript, conn_seeds = [1, 2, 3, 4, 5]
num_stims = 1      # number of stimulations per network; in manuscript, num_stims = 5


def sim_model():
    """Run the simulation for the specified model and return the output path."""
    from general_code.group_simulations import run_sim_group

    # Set up group options
    group_label = f'figS3C_' + 'r%d_' % ei_ratio + time.strftime("%Y%m%d_%H%M%S")
    group_options = {
        'max_cores': 50,
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
        'output_plots': False
    }

    # Set up group params
    n_b = int(20000 / ei_ratio)
    p_jb = 0.01 * ei_ratio / 4    
   
    d_num = 18
    group_params = {
        'n_stims': [num_stims],
        'conn_seed': [seed for seed in conn_seeds for _ in range(d_num * d_num)],
        'p_rc': list(np.round(np.linspace(0.00, 0.17, d_num), 3)) * d_num * len(conn_seeds),
        'p_ff': list(np.repeat(np.round(np.linspace(0.00, 0.17, d_num), 3), d_num)) * len(conn_seeds),
        'n_b': [n_b],
        'p_bb': [p_jb],
        'p_pb': [p_jb]
    }

    # Dynamically import the model's run_simulation function
    run_sim_group(group_options,
                  group_params, 
                  run_simulation)
    return os.path.join(os.getcwd(), 'outputs', group_label)


def plot_results(group_path=None):
    """Plot the results for figure S3C. If group_path is None, select interactively."""

    # Select the output directory if not provided
    if group_path is None:
        group_path = select_group_label_dir('outputs', 'figS3C_')

    # Generate pivot table
    get_replay_pivot(group_path)

    # Plot asymptotic speed
    group_plot(group_path, 'asy_speed',
               label_=r'Speed (1/ms)',
               vspan_=[0., 0.6],
               cbar_ticks_=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
               cbar_ticklabels_=['', '0.1', '', '0.3', '', '0.5', ''],
               square_coords=[(14, 0, '#7eb4ffff'), (7, 12, '#3cb371ff')],
               color_map='gist_heat')

    # Plot asymptotic width
    group_plot(group_path, 'asy_width',
               label_=r'FWHM (ms)',
               vspan_=[4.5, 10.5],
               cbar_ticks_=[5, 6, 7, 8, 9, 10],
               cbar_ticklabels_=['5', '', '7', '', '9', ''],
               square_coords=[(14, 0, '#7eb4ffff'), (7, 12, '#3cb371ff')],
               color_map='gist_heat_r')

    # Plot correlations
    get_correlations(group_path)

    print(f'Finished plots for {group_path}\n')


if __name__ == '__main__':
    # Default to 'sim' (auto-plot after). Explicit 'plot' runs plot-only.
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action not in ['sim', 'plot']:
            print("Invalid action. Valid options: sim, plot")
            sys.exit(1)
    else:
        action = 'sim'

    if action == 'sim':
        out_path = sim_model()
        plot_results(out_path)
    else:  # action == 'plot'
        plot_results()
    