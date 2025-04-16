from brian2 import *
import os
from my_code.simulations import run_sim_group
from population_models.rectangle import run_simulation
from my_code.plot_group_population import get_square_group

if __name__ == '__main__':
    group_options = {
        'save_test': False,
        'use_tex': True,
        'save_plots': True,
        'max_cores': 50,
        'n_tests': 1,
        'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'output_dir': os.getcwd() + '/outputs/fig3_E_',
        'output_plots': False
    }

    d_num = 49
    group_params = {
            'n_asb': [10],
            'w_rc': list(np.round(np.linspace(0.0, 1.2, d_num), 3)) * d_num,
            'w_ff': list(np.repeat(np.round(np.linspace(0.0, 1.2, d_num), 3), d_num)),
            'max_speed_inv': [1.5]
    }

    run_sim_group(group_options, group_params, run_simulation)

    get_square_group(group_options, n_asb=10, max_speed_inv=2.5,
                     max_cbar=1.5, plot_theory=True)