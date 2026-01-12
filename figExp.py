
import time
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from brian2 import nS, second, Hz, mV, pA
from general_code.group_simulations import run_sim_group
from spiking_networks.model3_delta import run_simulation

if __name__ == '__main__':

    # Set up group options
    group_label = f'figEXP_' + time.strftime("%Y%m%d_%H%M%S")
    group_options = {
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
        'output_plots': True,
    }

    # Set up group parameters
    group_params = {
        'p_rc': [0.00, 0.12],
        'p_ff': [0.14, 0.07],
        'n_stims': [1],
        'test_ai': [True],
        'n_e': [2500],
        'p_pe': [0.01],
        'rate_ext': [(5, Hz)],
        'syn_weight_pe': [(1.5, mV)],
        'syn_weight_pp': [(0.2, mV)],
        'curr_bg': [(0, pA)],
        'e_rest': [(-65, mV)],
        'v_reset': [(-75, mV)],
        'v_thres': [(-50, mV)],
        'g_leak': [(5, nS)],
    }

    run_sim_group(group_options,
                  group_params, 
                  run_simulation)