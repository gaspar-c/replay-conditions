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
from brian2 import nS, second
from general_code.group_simulations import run_sim_group
from general_code.aux_functions import select_group_label_dir
from spiking_networks.model2 import run_simulation
from spiking_networks.tests import fit_v_snapshot


ei_ratio = 4  # ratio of excitatory to inhibitory neurons
scale = 'p'   # scaling method for inhibitory synapses
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
        
        # Create plot using shared function
        fig_name = v_file.replace('_value.txt', '_replot')
        fit_params = fit_v_snapshot(v_snapshot, fig_name, log=None)
        
        # Extract metadata from filename for display
        basename = os.path.basename(v_file)
        print(f'\nProcessed: {basename}')
        if not np.isnan(fit_params['gauss_r2']):
            print(f"  R² = {fit_params['gauss_r2']:.3f}")
    
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
        # TestFitV already creates plots during simulation, so no need to call plot_results
    else:  # action == 'plot'
        plot_results()
