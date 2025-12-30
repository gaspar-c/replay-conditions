"""
figS3D.py
Generates Figure S3, panel D.

Run as 'python figS3D.py plot' to plot results from a previous run.
"""

import time
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from brian2 import nS, second
from scipy.optimize import curve_fit
from general_code.group_simulations import run_sim_group
from general_code.aux_functions import select_group_label_dir
from spiking_networks.model2 import run_simulation


def gaussian(x, amp, cen, wid):
    """Gaussian function for curve fitting."""
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


ei_ratio = 8  # ratio of excitatory to inhibitory neurons
scale = 'g_sqrt'   # scaling method for inhibitory synapses
stdp_time = 5 # duration of STDP training in seconds


def sim_model():
    """Run the simulation and return the output path."""
    
    # Set up group options
    group_label = f'figS3D_' + 'r%d_s%s_STDP%ds_' % (ei_ratio, scale, stdp_time) + time.strftime("%Y%m%d_%H%M%S")
    group_options = {
        'output_dir': os.getcwd() + '/outputs/',
        'group_label': group_label,
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
        'p_rc': [0.12, 0.10],
        'p_ff': [0.07, 0.10],
        'n_stims': [1],
        'test_ai': [True],
        'stdp_time': [(stdp_time,  second)],
        'n_b': [n_b],
        'p_bb': [p_jb],
        'p_pb': [p_jb],
        'g_bb': [(g_jb, nS)],
        'g_pb_init': [(g_jb, nS)],
    }

    run_sim_group(group_options,
                  group_params, 
                  run_simulation)
    
    return os.path.join(os.getcwd(), 'outputs', group_label)


def plot_results(group_path=None):
    """Plot the v_snapshot results. If group_path is None, select interactively."""
    
    # Select the output directory if not provided
    if group_path is None:
        group_path = select_group_label_dir('outputs', 'figS3D_')
    
    print(f'\nPlotting results from: {group_path}\n')
    
    # Find all v_snapshot value files
    v_files = sorted(glob.glob(os.path.join(group_path, '*_value.txt')))
    
    if not v_files:
        print(f"No v_snapshot files found in {group_path}")
        return
    
    print(f"Found {len(v_files)} v_snapshot files")
    
    for v_file in v_files:
        # Load the v_snapshot data
        v_snapshot = np.loadtxt(v_file)
        
        # Extract metadata from filename
        # Expected format: sim#_v{asb_idx}_{time}s_value.txt
        basename = os.path.basename(v_file)
        parts = basename.replace('_value.txt', '').split('_')
        
        # Create histogram
        counts, bin_edges = np.histogram(v_snapshot, bins=np.linspace(-60, -50, 41), density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate basic statistics
        v_mean = np.mean(v_snapshot)
        v_std = np.std(v_snapshot)
        
        # Try to fit Gaussian
        gauss_mean = np.nan
        gauss_std = np.nan
        gauss_u = np.nan
        gauss_x0 = np.nan
        gauss_r2 = np.nan
        fit_gauss = False
        
        try:
            fit_gauss_params, _ = curve_fit(gaussian, bin_centers, counts,
                                          p0=[np.max(counts), v_mean, v_std])
            fit_gauss = True
            gauss_height, gauss_mean, gauss_std = fit_gauss_params
            gauss_u = gauss_std * 4
            gauss_x0 = gauss_mean + 2 * gauss_std + 50
            
            # Calculate R²
            residuals = counts - gaussian(bin_centers, *fit_gauss_params)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((counts - np.mean(counts)) ** 2)
            gauss_r2 = 1 - (ss_res / ss_tot)
            
            print(f'\n{basename}:')
            print(f'  v (mean ± std) = ({v_mean:.3f} ± {v_std:.3f}) mV')
            print(f'  Gaussian μ = {gauss_mean:.3f} mV')
            print(f'  Gaussian σ = {gauss_std:.3f} mV')
            print(f'  U = {gauss_u:.3f} mV')
            print(f'  x0 = {gauss_x0:.3f} mV')
            print(f'  U - x0 = {gauss_u - gauss_x0:.3f} mV')
            print(f'  R² = {gauss_r2:.3f}')
            
        except (RuntimeError, ValueError) as e:
            print(f'\n{basename}: Could not fit Gaussian - {e}')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(2.5, 2))
        font_size = 8.3
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color='black')
        
        v_range = max(v_snapshot) - min(v_snapshot)
        x_values = np.linspace(min(bin_centers) - v_range * 0.05, max(bin_centers) + v_range * 0.05, 100)
        max_plot = -49.8
        
        if fit_gauss and gauss_r2 > 0.5:
            fig.text(0.92, 0.90, r'Gauss $R^2$ = %.2f' % gauss_r2, fontsize=font_size, va='top')
            fig.text(0.92, 0.83, r'- $\mu$ = %.2f mV' % gauss_mean, fontsize=font_size, va='top')
            fig.text(0.92, 0.76, r'- $\sigma$ = %.2f mV' % gauss_std, fontsize=font_size, va='top')
            fig.text(0.92, 0.69, r'- $U$ = %.2f mV' % gauss_u, fontsize=font_size, va='top')
            fig.text(0.92, 0.62, r'- $x_0$ = %.2f mV' % gauss_x0, fontsize=font_size, va='top')
            fig.text(0.92, 0.55, r'- $U - x_0$ = %.2f mV' % (gauss_u - gauss_x0), fontsize=font_size, va='top')
            ax.plot(x_values, gaussian(x_values, *fit_gauss_params), color='black', lw=2)
            ax.plot(x_values, gaussian(x_values, *fit_gauss_params), color='darkgray', lw=1.5, alpha=0.95)
            ax.axvspan(gauss_mean - 2*gauss_std, gauss_mean + 2*gauss_std, 
                       alpha=0.2, color='lightblue', zorder=0)
            if gauss_mean + 2*gauss_std > max_plot:
                max_plot = gauss_mean + 2*gauss_std + 0.2
        
        ax.set_xlim([-61, max_plot])
        ax.set_ylim([0, max(counts) * 1.1])
        ax.axvline(-50, color='black', lw=1.5, ls='--')
        ax.set_xticks([-60, -55, -50], labels=['-60', '', '-50'], fontsize=font_size)
        ax.set_xlabel(r'$v$ (mV)', fontsize=font_size)
        ax.set_ylabel('count', fontsize=font_size)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, colors='black', labelsize=font_size)
        
        # Save plot
        fig_name = v_file.replace('_value.txt', '_replot')
        fig.savefig(fig_name + '.png', dpi=300, bbox_inches='tight')
        fig.savefig(fig_name + '.svg', bbox_inches='tight')
        plt.close(fig)
    
    print(f'\nFinished plotting {len(v_files)} files from {group_path}\n')


if __name__ == '__main__':
    # Default to 'sim'. Explicit 'plot' runs plot-only.
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action not in ['sim', 'plot']:
            print("Invalid action. Valid options: sim, plot")
            sys.exit(1)
    else:
        action = 'sim'
    
    if action == 'sim':
        out_path = sim_model()
        # Optionally plot results after simulation
        # plot_results(out_path)
    else:  # action == 'plot'
        plot_results()
