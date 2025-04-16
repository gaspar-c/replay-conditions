from brian2 import *
import os
from my_code.simulations import run_sim_group
from population_models.rectangle import run_simulation

if __name__ == '__main__':
    group_options = {
        'save_test': False,
        'use_tex': True,
        'save_plots': True,
        'max_cores': 3,
        'n_tests': 1,
        'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'output_dir': os.getcwd() + '/outputs/fig3_D_',
        'output_plots': True
    }

    # 2D R-F simulations
    group_params = {
        'n_asb': [10],
        'w_ff': [1.2, 0.60, 0.6],
        'w_rc': [0.0, 0.60, 0.2],
        'max_speed_inv': [1.5],
    }

    run_sim_group(group_options, group_params, run_simulation)
