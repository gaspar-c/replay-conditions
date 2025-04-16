import time
import os
from my_code.simulations import run_sim_group


from spiking_models.model2 import run_simulation


if __name__ == '__main__':
    group_options = {
        'use_tex': True,
        'max_cores': 2,
        'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'output_dir': os.getcwd() + '/outputs/fig1_model2_BC_',
        'output_plots': True
    }

    group_params = {
        'n_stims': [1],
        'p_rc': [0.00, 0.12],
        'p_ff': [0.14, 0.07],
        'v_time': [19.6, 48.5],
    }

    run_sim_group(group_options, group_params, run_simulation)

