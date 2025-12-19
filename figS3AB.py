"""
figS3AB.py
Generates Figure S3, panels A and B.
"""

import time
import os
import numpy as np
from brian2 import nS
from general_code.aux_functions import get_model_number
from general_code.group_simulations import run_sim_group
from spiking_networks.model2 import run_simulation


ei_ratio = 8  # ratio of excitatory to inhibitory neurons
scale = 'p'

if __name__ == '__main__':

    # Set up group options
    group_options = {
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': f'figS3AB_' +
          'r%d_scale_%s_' % (ei_ratio, scale) +
          time.strftime("%Y%m%d_%H%M%S"),
        'output_plots': True,
    }
    
    n_b = int(20000 / ei_ratio)
    if scale == 'p':
        p_jb = 0.01 * ei_ratio / 4    
        g_jb = 0.40
    elif scale == 'g':
        p_jb = 0.01
        g_jb = 0.40 * ei_ratio / 4
    elif scale == 'g_sqrt':
        p_jb = 0.01
        g_jb = 0.40 * np.sqrt(ei_ratio / 4)
   
    # Set up group parameters
    group_params = {
        'p_rc': [0.00, 0.12],
        'p_ff': [0.14, 0.07],
        'n_b': [n_b],
        'p_bb': [p_jb],
        'p_pb': [p_jb],
        'g_bb': [(g_jb, nS)],
        'g_pb_init': [(g_jb, nS)],
        'v_time': [19.6, 48.5]
    }

    run_sim_group(group_options,
                  group_params, 
                  run_simulation)
