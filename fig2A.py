"""
fig2A.py
Generates Figure 2, panel A.

To reproduce manuscript results, manually change tau_l_pp to 0.1, 1, 2, or 4.

p_rc and p_ff are varied from 0.00 to 0.17 in 18 steps each. 
Each (p_ff, p_rc) pair is run with multiple random connectivity seeds.
Thus, number of simulated networks is 18 x 18 x len(conn_seeds). 
Each network is stimulated num_stims times.

For quicker simulation times, this file uses as default conn_seeds = [1] and num_stims = 1.
In the manuscript, we used num_stims = 5 and conn_seeds = [1, 2, 3, 4, 5].

Run as 'python fig2A.py plot' to plot results from a previous run.
"""

import time
import os
import sys
import numpy as np
from brian2 import ms
from spiking_networks.group_replay_analysis import get_replay_pivot, group_plot
from general_code.aux_functions import select_group_label_dir


tau_l_pp = 1        # synaptic latency (ms)
conn_seeds = [1]    # random seeds; in manuscript, conn_seeds = [1, 2, 3, 4, 5]
num_stims = 1       # number of stimulations per network; in manuscript, num_stims = 5


def sim_model():
    """Run the simulation for figure 2A and return the output path."""
    from general_code.group_simulations import run_sim_group
    from spiking_networks.model3 import run_simulation

    group_label = 'fig2A_%.1fms_' % tau_l_pp + time.strftime("%Y%m%d_%H%M%S")

    group_options = {
        'max_cores': 50,
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
        'output_plots': False
    }

    d_num = 18
    group_params = {
        'n_stims': [num_stims],
        'conn_seed': [seed for seed in conn_seeds for _ in range(d_num * d_num)],
        'p_rc': list(np.round(np.linspace(0.00, 0.17, d_num), 3)) * d_num * len(conn_seeds),
        'p_ff': list(np.repeat(np.round(np.linspace(0.00, 0.17, d_num), 3), d_num)) * len(conn_seeds),
        'tau_l_pp': [(tau_l_pp, ms)]
    }

    run_sim_group(group_options, group_params, run_simulation)
    
    return os.path.join(os.getcwd(), 'outputs', group_label)


def plot_results(group_path=None):
    """Plot the results for figure 2A. If group_path is None, select interactively."""
    if group_path is None:
        group_path = select_group_label_dir('outputs', 'fig2A_')
    
    middle = group_path.split('fig2A_')[1]   
    num_str = middle.split('ms')[0]           
    tau_l_pp = float(num_str) 

    get_replay_pivot(group_path)

    tau_m = 20
    tau_d = 2
    time_step = 1/((1/tau_d) + (1/tau_m)) + tau_l_pp
    dist_width = 12.0

    group_plot(group_path, 'asy_speed',
               label_=r'Speed (1/ms)',
               cond3=[0.00, 0.20, 0.30], time_step=time_step, dist_width=dist_width,
               vspan_=[0., 0.4],
               cbar_ticks_=[0.0, 0.1, 0.2, 0.3, 0.4],
               cbar_ticklabels_=['0.0', '', '0.2', '', '0.4'],
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
