"""
fig4BD.py
Generates Figure 4, panels B and D.
Prompts user to select which model to run ('gaussian' or 'rectangle').
"""

import time
import os
import sys
from general_code.aux_functions import get_model_shape
from general_code.group_simulations import run_sim_group


VALID_MODELS = ['gaussian', 'rectangle']


if __name__ == '__main__':
     
    """Run simulations for figure 4B/4D depending on model."""

    model_name = get_model_shape()
    if model_name == 'gaussian':
        from population_model.model_gauss import run_simulation
        label_prefix = 'fig4B_'
    elif model_name == 'rectangle':
        from population_model.model_rectangle import run_simulation
        label_prefix = 'fig4D_'
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {VALID_MODELS}.")

    group_options = {
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': label_prefix + time.strftime("%Y%m%d_%H%M%S"),
        'output_plots': True
    }

    group_params = {
        'n_asb': [10],
        'steps_per_asb': [1.5], # number of time steps to simulate = steps_per_asb * n_asb 
        'w_rc': [0.0, 0.6, 0.2],
        'w_ff': [1.2, 0.6, 0.6],
    }

    run_sim_group(group_options, group_params, run_simulation)
