from brian2 import *
import time
import os
from my_code.simulations import run_sim_group
from my_code.plot_group_spiking import get_replay_pivot, group_plot
from spiking_models.model3_no_leak import run_simulation


group_options = {
                  'use_tex': True,
                  'max_cores': 50,
                  # 'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
                  'time_stamp': '20250416_152730',
                  'output_dir': os.getcwd() + '/outputs/fig4_B_',
                  'output_plots': False
                 }


if __name__ == '__main__':

    sigma = 0.5  # mV
    mu = -51.9   # mV
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
        'init_normal_sigma': [(sigma, mV)],
        'init_normal_mean': [(mu, mV)]
    }

    run_sim_group(group_options, group_params, run_simulation)

    group_path = group_options['output_dir'] + group_options['time_stamp']
    get_replay_pivot(group_path, 'p_ff', 'p_rc', qual_thres=0.8,
                     act_low_thres=0.90, act_up_thres=1.10,
                     )

    w = 0.10 * 2 * (- mu) / 200
    x0 = mu + 50 + 2 * sigma
    U = 4 * sigma
    cond2 = (U - x0) / (500*w) * 100
    cond3 = - x0 / (500*w) * 100

    group_plot(group_path, 'asy_speed',
               'p_ff', 'p_rc',
               r'$p_f$ (\%)', r'$p_r$ (\%)',
               label_=r'Speed (1/ms)',
               scale_x=100, scale_y=100,
               skip_xlabel=4, skip_ylabel=4,
               log_scale=False,
               theor_plot=False,
               vspan_=[0., 0.7],
               cbar_ticks_=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
               cbar_ticklabels_=['', '0.1', '', '0.3', '', '0.5', '', '0.7'],
               color_map='gist_heat',
               cond2=cond2,
               cond3=cond3)
