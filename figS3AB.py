"""
figS3AB.py
Generates Figure S3, panels A and B.

To reproduce manuscript results, manually change w_ee to 40, 50, or 60.
"""

import time
import os
from brian2 import mV
from general_code.group_simulations import run_sim_group
from spiking_networks.model3_delta import run_simulation

w_ee = 40 # EPSP (microV)

if __name__ == '__main__':

    group_options = {
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': 'figS2AB_w%.0f_' % w_ee + time.strftime("%Y%m%d_%H%M%S"),
        'output_plots': True,
    }

    group_params = {
        'n_stims': [1],
        'p_rc': [0.00, 0.16],
        'p_ff': [0.14, 0.04],
        'syn_weight_pp': [(w_ee * 1e-3, mV)],
    }

    run_sim_group(group_options, group_params, run_simulation)
