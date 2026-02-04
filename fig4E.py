"""
fig4E.py
Generates Figure 4, panel E.

To reproduce manuscript results, manually change n_asb to 10, 50, or 200.

Run as 'python fig4E.py plot' to plot results from a previous run.
"""

import time
import os
import sys
import numpy as np
from population_model.group_replay_analysis import get_replay_pivot, group_plot
from general_code.aux_functions import select_group_label_dir


n_asb = 200  # number of assemblies (q)

def sim_model():
    """Run the simulation for figure 4E and return the output path."""
    from general_code.group_simulations import run_sim_group
    from population_model.model_rectangle import run_simulation
    
    group_label = 'fig4E_q%d_' % n_asb + time.strftime("%Y%m%d_%H%M%S")
    
    group_options = {
        'max_cores': 50,
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
        'output_plots': False
    }

    d_num = 49
    group_params = {
        'n_asb': [n_asb],
        'w_rc': list(np.round(np.linspace(0.0, 1.2, d_num), 3)) * d_num,
        'w_ff': list(np.repeat(np.round(np.linspace(0.0, 1.2, d_num), 3), d_num)),
        'steps_per_asb': [1.5]
    }

    run_sim_group(group_options, group_params, run_simulation)
    
    return os.path.join(os.getcwd(), 'outputs', group_label)


def plot_results(group_path=None):
    """Plot the results for figure 4E."""
    if group_path is None:
        group_path = select_group_label_dir('outputs', 'fig4E_')
    
    middle = group_path.split('fig4E_q')[1]   
    num_str = middle.split('_')[0]           
    n_asb = int(num_str) 

    get_replay_pivot(group_path, n_asb)

    group_plot(group_path, 'asy_speed', n_asb=n_asb, plot_theory=True, max_cbar=1.5)

    print('finished plots for %s' % group_path)


if __name__ == '__main__':
    # Parse command-line arguments: default is 'sim' (with auto-plot), explicit 'plot' runs only plot
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action not in ['sim', 'plot']:
            print("Invalid action. Valid options: sim, plot")
            sys.exit(1)
    else:
        action = 'sim'  # Default action

    if action == 'sim':
        group_path = sim_model()
        plot_results(group_path)
    else:  # action == 'plot'
        plot_results()
