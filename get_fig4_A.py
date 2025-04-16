from brian2 import *
import time
import os
from my_code.simulations import run_sim_group
from spiking_models.model3 import run_simulation
from my_code.plot_group_spiking import get_replay_pivot, get_speed_lines

if __name__ == '__main__':

    group_options = {
        'use_tex': True,
        'max_cores': 50,
        # 'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'time_stamp': '20250416_143021',
        'output_dir': os.getcwd() + '/outputs/fig4_A_',
        'output_plots': False
    }

    tau_l_pp = 2  # ms
    d_num = 18
    group_params = {
        'n_stims': [1],
        # 'conn_seed': ([1] * d_num * d_num +
        #               [2] * d_num * d_num +
        #               [3] * d_num * d_num +
        #               [4] * d_num * d_num +
        #               [5] * d_num * d_num),
        'p_rc': list(np.round(np.linspace(0.00, 0.17, d_num), 3)) * d_num * 1,
        'p_ff': list(np.repeat(np.round(np.linspace(0.00, 0.17, d_num), 3), d_num)) * 1,
        'tau_l_pp': [(tau_l_pp, ms)]
    }

    # run_sim_group(group_options, group_params, run_simulation)

    group_path = group_options['output_dir'] + group_options['time_stamp']
    get_replay_pivot(group_path, 'p_ff', 'p_rc', qual_thres=0.8,
                     act_low_thres=0.90, act_up_thres=1.10,
                     )

    tau_m = 20
    tau_d = 2
    time_step = 1/((1/tau_d) + (1/tau_m)) + tau_l_pp
    dist_width = 12.0
    get_speed_lines(group_options, time_step, dist_width, speed_lims=[None, 0.20, 0.30])
