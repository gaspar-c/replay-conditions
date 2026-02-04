"""
fig1BC.py
Generates Figure 1, panels B and C for models 1, 2, and 3.
Prompts user to select which model to run.
"""

import time
import os
import sys
from general_code.aux_functions import get_model_number
from general_code.group_simulations import run_sim_group


# Model-specific parameters (times to plot v snapshot)
MODEL_PARAMS = {
    1: [20.0, 41.0],
    2: [19.6, 48.5],
    3: [17.5, 29.2]
}


if __name__ == '__main__':
    # Get model number from user or command-line argument
    if len(sys.argv) > 1:
        try:
            model_num = int(sys.argv[1])
            if model_num not in MODEL_PARAMS:
                print(f"Invalid model number: {model_num}. Valid options: 1, 2, 3")
                sys.exit(1)
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Please provide a number (1, 2, or 3)")
            sys.exit(1)
    else:
        model_num = get_model_number(valid_models=list(MODEL_PARAMS.keys()))

    # Set up group options
    group_options = {
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': f'fig1BC_model{model_num}' + '_' + time.strftime("%Y%m%d_%H%M%S"),
        'output_plots': True,
    }
    
    # Set up group parameters
    group_params = {
        'p_rc': [0.00, 0.12],
        'p_ff': [0.14, 0.07],
        'v_time': MODEL_PARAMS[model_num]
    }

    # Dynamically import the model's run_simulation function
    run_sim_group(group_options,
                  group_params, 
                  __import__(f'spiking_networks.model{model_num}',
                             fromlist=['run_simulation']
                            ).run_simulation)
