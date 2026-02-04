"""
fig2B.py
Generates Figure 2, panel B.

To reproduce manuscript results, manually change
mu to -50.9, -51.9, or -52.9 and sigma to 0.0, 0.5, or 1.0.

p_rc and p_ff are varied from 0.00 to 0.17 in 18 steps each. 
Each (p_ff, p_rc) pair is run with multiple random connectivity seeds.
Thus, number of simulated networks is 18 x 18 x len(conn_seeds). 
Because there is no leak or reset, each network is stimulated only once.

For quicker simulation times, this file uses as default conn_seeds = [1].
In the manuscript, we used conn_seeds = [1, 2, 3, 4, 5].

Run as 'python fig2B.py plot' to plot results from a previous run.
"""

import time
import os
import sys
import numpy as np
from brian2 import mV
from spiking_networks.group_replay_analysis import get_replay_pivot, group_plot
from general_code.aux_functions import select_group_label_dir


mu = -52.9          # mean of gaussian distribution (mV)
sigma = 1.0         # standard deviation of gaussian distribution (mV)
conn_seeds = [1]    # number of stimulations per network; in manuscript, num_stims = 5

def sim_model():
    """Run the simulation for figure 2B and return the output path."""
    from general_code.group_simulations import run_sim_group
    from spiking_networks.model3_no_leak import run_simulation

    group_label = 'fig2B_mu%.1fmV_sigma%.1fmV_' % (mu, sigma) + time.strftime("%Y%m%d_%H%M%S")

    group_options = {
        'max_cores': 50,
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
        'output_plots': False
    }

    d_num = 18
    group_params = {
        'conn_seed': [seed for seed in conn_seeds for _ in range(d_num * d_num)],
        'p_rc': list(np.round(np.linspace(0.00, 0.17, d_num), 3)) * d_num * len(conn_seeds),
        'p_ff': list(np.repeat(np.round(np.linspace(0.00, 0.17, d_num), 3), d_num)) * len(conn_seeds),
        'init_gaussian_sigma': [(sigma, mV)],
        'init_gaussian_mean': [(mu, mV)]
    }

    run_sim_group(group_options, group_params, run_simulation)
    
    return os.path.join(os.getcwd(), 'outputs', group_label)


def plot_results(group_path=None):
    """Plot the results for figure 2B."""
    if group_path is None:
        group_path = select_group_label_dir('outputs', 'fig2B_')

    # Find mu and sigma from path 
    after_prefix = group_path.split('fig2B_')[1]   
    after_mu     = after_prefix.split('mu')[1]  
    mu_str      = after_mu.split('mV')[0]      
    after_sigma = after_mu.split('sigma')[1]   
    sigma_str   = after_sigma.split('mV')[0]   
    mu    = float(mu_str)
    sigma = float(sigma_str)

    # Calculate conditions 1 and 2
    w = 0.10 * 2 * (- mu) / 200
    x0 = mu + 50 + 2 * sigma
    U = 4 * sigma
    cond1 = (U - x0) / (500*w) * 100
    cond2 = - x0 / (500*w) * 100

    get_replay_pivot(group_path)

    group_plot(group_path, 'asy_speed',
               label_=r'Speed (1/ms)',
               cond1=cond1,
               cond2=cond2,
               vspan_=[0., 0.7],
               cbar_ticks_=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
               cbar_ticklabels_=['', '0.1', '', '0.3', '', '0.5', '', '0.7'],
               color_map='gist_heat')

    print('finished plots for %s' % group_path)


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
