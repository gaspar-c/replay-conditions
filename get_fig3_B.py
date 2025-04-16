from brian2 import *
import os
from my_code.simulations import run_sim_group
from population_models.gauss import run_simulation



if __name__ == '__main__':
    group_options = {
        'save_test': False,
        'use_tex': True,
        'save_plots': True,
        'max_cores': 3,
        'n_tests': 1,
        'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'output_dir': os.getcwd() + '/outputs/fig3_B_',
        'output_plots': True
    }

    group_params = {
        'n_asb': [10],
        'max_speed_inv': [1.5],
        'w_rc': [0.0, 0.6, 0.2],
        'w_ff': [1.0, 0.6, 0.6],
    }

    run_sim_group(group_options, group_params, run_simulation)


