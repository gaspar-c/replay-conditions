import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, rc, patches
from matplotlib.colors import LogNorm
import scipy.optimize
import pandas as pd
from my_code.discrete_model import simplified_theory

rc('text', usetex=True)
rc('mathtext', fontset='stix')
rc('font', family='sans-serif')

if __name__ == '__main__':

    # plot results
    fig_height = 1.4
    fig_width = fig_height * 0.9
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    color_map = 'gist_heat_r'
    cmap = plt.get_cmap(color_map)

    # Create a new range of x and y values with the defined step size
    range_q = np.arange(10, 500 + 2, 2)
    range_f = np.arange(0.01, 1.2, 0.001)
    fixed_r = np.array([0.4])

    for speed_inv in [1.1, 1.2, 1.3, 1.4, 1.5, 0.0]:
        theor_thres = np.zeros(len(range_q))
        k = 0
        for q_ in range_q:
            if speed_inv > 0:
                t_step = np.floor(speed_inv * q_)
            else:
                t_step = np.nan
            theor_check = simplified_theory(range_f, fixed_r, q_, t_step, init_j=1.0)
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
        # ax.plot(range_q, theor_thres, color='black', lw=2)
        if speed_inv > 0:
            v_max = 1 / 2.5
            line_color = cmap((1 / speed_inv - 1.0) / (v_max - 1.0))
        else:
            line_color = 'black'
        ax.plot(range_q, theor_thres, '.', color=line_color, ms=2, markeredgewidth=0.0)
        ax.axhline(1/v, color=line_color, ls=(0, (5, 4)), lw=1)

    # edit axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=3, width=1, direction='out', labelsize=font_size)
    ax.set_xticks([10, 125, 250, 375, 500])
    ax.set_xticklabels(['10', '125', '250', '375', '500'])
    ax.set_ylim([0.55, 0.9])
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9'])
    ax.set_xlabel('Sequence Length', fontsize=font_size)
    ax.set_ylabel(r'Feedforward ($F$)', fontsize=font_size)

    plt.savefig('outputs/fig3_F.png', dpi=600, bbox_inches='tight')
    plt.savefig('outputs/fig3_F.svg', dpi=600, bbox_inches='tight')
