from brian2 import *
import time
import os
from my_code.simulations import run_sim_group


from spiking_models.model3_delta import run_simulation




if __name__ == '__main__':
    group_options = {
        'use_tex': True,
        'max_cores': 2,
        'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'output_dir': os.getcwd() + '/outputs/figA1_AB_',
        'output_plots': True,
    }

    group_params = {
        'p_rc': [0.00, 0.16],
        'p_ff': [0.14, 0.04],
        'syn_weight_pp': [(0.06, mV)],
        'tau_l_pp': [(3, ms)],
    }

    run_sim_group(group_options, group_params, run_simulation)