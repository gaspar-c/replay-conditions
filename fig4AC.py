"""
fig4AC.py
Generates Figure 4, panels A and C.
Prompts user to select which model to run ('gaussian' or 'rectangle').

Run as 'python fig4AC.py plot' to plot results from a previous run.
"""

import time
import os
import sys
import numpy as np
from population_model.group_replay_analysis import get_replay_pivot, group_plot, get_correlations
from general_code.aux_functions import select_group_label_dir, get_model_shape


VALID_MODELS = ['gaussian', 'rectangle']


def sim_model(model_name):
    """Run simulations for figure 4A/4C depending on model and return the output path."""
    from general_code.group_simulations import run_sim_group

    if model_name == 'gaussian':
        from population_model.model_gauss import run_simulation
        label_prefix = 'fig4A_'
    elif model_name == 'rectangle':
        from population_model.model_rectangle import run_simulation
        label_prefix = 'fig4C_'
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {VALID_MODELS}.")

    group_label = label_prefix + time.strftime("%Y%m%d_%H%M%S")

    group_options = {
        'max_cores': 50,
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
        'output_plots': False
    }

    # 2D R-F simulations
    d_num = 49
    group_params = {
        'n_asb': [10],
        'w_rc': list(np.round(np.linspace(0.0, 1.2, d_num), 3)) * d_num,
        'w_ff': list(np.repeat(np.round(np.linspace(0.0, 1.2, d_num), 3), d_num)),
    }

    run_sim_group(group_options, group_params, run_simulation)
    return os.path.join(os.getcwd(), 'outputs', group_label)


def plot_results(group_path=None):
    """Plot results for figure 4A/4C. If group_path is None, select folder interactively."""
    if group_path is None:
        group_path = select_group_label_dir('outputs', 'fig4[AC]_')
    
    get_replay_pivot(group_path, 10)

    group_plot(group_path, 'asy_speed', n_asb=10, draw_levels=True)

    group_plot(group_path, 'asy_width', n_asb=10, draw_levels=True)

    get_correlations(group_path, 10)

    print('finished plots for %s' % group_path)


if __name__ == '__main__':
    # Default to 'sim'; explicit 'plot' runs plot-only with interactive selection
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action not in ['sim', 'plot']:
            print("Invalid action. Valid options: sim, plot")
            sys.exit(1)
    else:
        action = 'sim'

    if action == 'sim':
        model_name = get_model_shape()
        group_path = sim_model(model_name)
        plot_results(group_path)
    else:
        plot_results()
