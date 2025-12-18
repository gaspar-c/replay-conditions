"""
fig4F.py
Generates Figure 4, panel F.
"""

import time
import os
import numpy as np
from matplotlib import pyplot as plt, rc
from population_model.theory import cond_fin_q_fin_t, cond_fin_q_inf_t

rc('mathtext', fontset='stix')
rc('font', family='sans-serif')

if __name__ == '__main__':
    # Fixed recurrent strength
    r_value = 0.4
    
    # Create timestamped output folder
    output_dir = os.getcwd() + '/outputs/'
    group_label = 'fig4F_' + time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, group_label)
    os.makedirs(output_path, exist_ok=True)

    # Create a new range of x and y values with the defined step size
    range_q = np.arange(10, 500 + 2, 2)
    range_f = np.arange(0.01, 1.2, 0.001)
    fixed_r = np.array([r_value])

    # plot results
    fig_height = 1.4
    fig_width = fig_height * 0.9
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    color_map = 'gist_heat_r'
    cmap = plt.get_cmap(color_map)
    
    # Sweep inverse speeds (finite-time) plus 0 for infinite-time case
    for speed_inv in [1.1, 1.2, 1.3, 1.4, 1.5, 0.0]:
        theor_thres = np.zeros(len(range_q))
        k = 0
        for q_ in range_q:
            if speed_inv > 0:
                theor_check = cond_fin_q_fin_t(range_f, fixed_r, q_, np.floor(speed_inv * q_))
            else:
                theor_check = cond_fin_q_inf_t(range_f, fixed_r, q_)
            
            theor_true = np.argwhere(theor_check == True)
            theor_thres[k] = range_f[np.min(theor_true[:, 0])]
            k += 1

        ff_2d = range_f[:, None]
        rc_2d = fixed_r[None, :]
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # Ignore division by zero
            v_inf = 1 / (1 - rc_2d)
            if speed_inv > 0:
                speed_val = 1 / speed_inv
                v_small = (1/speed_val) * ((rc_2d/(1-speed_val)) ** (1/speed_val - 1))
            else:
                speed_val = np.nan
                v_small = v_inf
        
        v = np.where(rc_2d < 1 - speed_val, v_inf, v_small)[0]
        
        if speed_inv > 0:
            v_max = 1 / 2.5
            line_color = cmap((1 / speed_inv - 1.0) / (v_max - 1.0))
        else:
            line_color = 'black'
        
        # Scatter minimal F threshold vs q and overlay regime line
        ax.plot(range_q, theor_thres, '.', color=line_color, ms=2, markeredgewidth=0.0)
        ax.axhline(1/v, color=line_color, ls=(0, (5, 4)), lw=1)

    # edit axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, width=1, direction='out', labelsize=font_size)
   
    ax.set_xticks([10, 125, 250, 375, 500])
    ax.set_xticklabels(['10', '125', '250', '375', '500'])
    ax.set_xlabel('Sequence Length', fontsize=font_size)

    ax.set_ylim([0.55, 0.9])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9'])
    ax.set_ylabel(r'Feedforward ($F$)', fontsize=font_size)

    plt.savefig(os.path.join(output_path, 'fig4F.png'), dpi=300, bbox_inches='tight')
    print(f'Figure saved to {output_path}')
