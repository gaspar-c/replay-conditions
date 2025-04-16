from brian2 import *
import time
import os
from my_code.simulations import run_sim_group
from my_code.plot_group_spiking import get_correlations, group_plot, get_replay_pivot

from spiking_models.model2 import run_simulation



if __name__ == '__main__':
    group_options = {
        'use_tex': True,
        'max_cores': 50,
        'time_stamp': time.strftime("%Y%m%d_%H%M%S"),
        'output_dir': os.getcwd() + '/outputs/fig1_model2_D_',
        'output_plots': False
    }

    d_num = 18
    group_params = {
        # 'conn_seed': ([1] * d_num * d_num +
        #               [2] * d_num * d_num +
        #               [3] * d_num * d_num +
        #               [4] * d_num * d_num +
        #               [5] * d_num * d_num),
        'p_rc': list(np.round(np.linspace(0.00, 0.17, d_num), 3)) * d_num * 1,
        'p_ff': list(np.repeat(np.round(np.linspace(0.00, 0.17, d_num), 3), d_num)) * 1,
    }

    run_sim_group(group_options, group_params, run_simulation)

    group_path = group_options['output_dir'] + group_options['time_stamp']
    get_replay_pivot(group_path, 'p_ff', 'p_rc', qual_thres=0.8,
                     act_low_thres=0.90, act_up_thres=1.10,
                     )

    group_plot(group_path, 'asy_speed',
               'p_ff', 'p_rc',
               r'$p_f$ (\%)', r'$p_r$ (\%)',
               label_=r'Speed (1/ms)',
               scale_x=100, scale_y=100,
               skip_xlabel=4, skip_ylabel=4,
               log_scale=False,
               theor_plot=False,
               vspan_=[0., 0.6],
               cbar_ticks_=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
               cbar_ticklabels_=['', '0.1', '', '0.3', '', '0.5', ''],
               square_coords=[(14, 0, '#7eb4ffff'), (7, 12, '#3cb371ff')],
               color_map='gist_heat')

    group_plot(group_path, 'asy_width',
               'p_ff', 'p_rc',
               r'$p_f$ (\%)', r'$p_r$ (\%)',
               label_=r'FWHM (ms)',
               scale_x=100, scale_y=100,
               skip_xlabel=4, skip_ylabel=4,
               log_scale=False,
               theor_plot=False,
               vspan_=[4.5, 10.5],
               cbar_ticks_=[5, 6, 7, 8, 9, 10],
               cbar_ticklabels_=['5', '', '7', '', '9', ''],
               square_coords=[(14, 0, '#7eb4ffff'), (7, 12, '#3cb371ff')],
               color_map='gist_heat_r')

    get_correlations(group_options)
