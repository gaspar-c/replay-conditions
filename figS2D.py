"""
figS2D.py
Generates Figure S2, panel D.

Run as 'python figS2D.py plot' to plot results from previous runs.
"""

import time
import os
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from brian2 import nS, second
from scipy.optimize import curve_fit
from general_code.group_simulations import run_sim_group
from spiking_networks.model2 import run_simulation

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

ei_ratio = 8  # ratio of excitatory to inhibitory neurons
scale = 'p'   # scaling method for inhibitory synapses
stdp_time = 300 # duration of STDP training in seconds


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


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


def plot_results():
    """Overlay v-snapshot histograms and fits from three runs with different ei_ratio.

    Prompts for three ei_ratio values that match the `scale` and `stdp_time`
    constants at the top of this file. For each selected ratio, the most
    recently modified matching output folder is used. Only v-snapshot files
    present in all three folders are plotted (shared basenames).
    """

    output_root = os.path.join(os.getcwd(), 'outputs')
    
    # Map ei_ratio -> list of dirs, pick latest per ratio
    ratio_to_dir = {}
    
    # For r4, match only stdp_time (any scale)
    r4_pattern = f'figS3D_r4_s*_STDP{stdp_time}s_*'
    r4_dirs = sorted(glob.glob(os.path.join(output_root, r4_pattern)))
    if r4_dirs:
        ratio_to_dir[4] = r4_dirs
    
    # For r2 and r8, match both scale and stdp_time
    base_pattern = f'figS3D_r*_s{scale}_STDP{stdp_time}s_*'
    candidate_dirs = sorted(glob.glob(os.path.join(output_root, base_pattern)))
    
    for d in candidate_dirs:
        try:
            name_part = os.path.basename(d)
            ratio_str = name_part.split('_')[1]  # r{ei}
            if ratio_str.startswith('r'):
                ei_val = int(ratio_str[1:])
            else:
                continue
            # Skip r4 since we already added it above
            if ei_val == 4:
                continue
            if ei_val not in ratio_to_dir:
                ratio_to_dir[ei_val] = []
            ratio_to_dir[ei_val].append(d)
        except Exception:
            continue

    print("Available ei_ratio values for this scale/stdp_time:")
    for r in sorted(ratio_to_dir.keys()):
        print(f"  r{r}: {len(ratio_to_dir[r])} runs")

    # Use fixed ratios in the order 4, 2, 8
    selected_ratios = [4, 2, 8]

    selected_dirs = []
    for ratio in selected_ratios:
        if ratio not in ratio_to_dir:
            if ratio == 4:
                print(f"No runs found for ei_ratio={ratio} with stdp_time={stdp_time}s (any scale)")
            else:
                print(f"No runs found for ei_ratio={ratio} with scale={scale} and stdp_time={stdp_time}s")
            return
        dir_list = sorted(ratio_to_dir[ratio], key=os.path.getmtime, reverse=True)
        # Only prompt if there are more than one run for this ratio
        if len(dir_list) > 1:
            print(f"\nMultiple runs found for r{ratio} (newest first):")
            for i, d in enumerate(dir_list):
                print(f"  [{i}] {d}")
            sel = input("Select index for r%d (blank for newest): " % ratio).strip()
            if sel == "":
                chosen_dir = dir_list[0]
            else:
                try:
                    idx = int(sel)
                    if idx < 0 or idx >= len(dir_list):
                        print("Index out of range; aborting.")
                        return
                    chosen_dir = dir_list[idx]
                except ValueError:
                    print("Invalid index; aborting.")
                    return
        else:
            chosen_dir = dir_list[0]
        selected_dirs.append((ratio, chosen_dir))

    print("\nUsing directories:")
    for ratio, d in selected_dirs:
        print(f"  r{ratio}: {d}")

    # Determine common v_snapshot files by basename
    dir_files = []
    for _, d in selected_dirs:
        files = [os.path.basename(f) for f in glob.glob(os.path.join(d, '*_value.txt'))]
        dir_files.append(set(files))

    common_files = set.intersection(*dir_files) if dir_files else set()
    if not common_files:
        print("No common v_snapshot files across the selected runs.")
        return

    # Group files by sim number (sim1, sim2, etc.) and other identifiers
    # Files are like: sim1_v10_300.000s_value.txt, sim2_v10_300.000s_value.txt, etc.
    sim_groups = {}
    for basename in common_files:
        # Extract sim number if present
        if basename.startswith('sim'):
            parts = basename.split('_', 1)
            sim_num = parts[0]  # e.g., 'sim1', 'sim2'
            if len(parts) > 1:
                rest = parts[1]  # e.g., 'v10_300.000s_value.txt'
            else:
                rest = basename
        else:
            sim_num = 'other'
            rest = basename
        
        if sim_num not in sim_groups:
            sim_groups[sim_num] = []
        sim_groups[sim_num].append(basename)
    
    # If multiple sim groups found, prompt user
    sim_keys = sorted(sim_groups.keys())
    if len(sim_keys) > 1:
        print(f"\nMultiple simulation groups found:")
        for i, sim_key in enumerate(sim_keys):
            print(f"  [{i}] {sim_key} ({len(sim_groups[sim_key])} files)")
        sel = input("Select simulation group to plot (blank for all): ").strip()
        if sel == "":
            files_to_plot = common_files
        else:
            try:
                idx = int(sel)
                if idx < 0 or idx >= len(sim_keys):
                    print("Index out of range; aborting.")
                    return
                files_to_plot = sim_groups[sim_keys[idx]]
            except ValueError:
                print("Invalid index; aborting.")
                return
    else:
        files_to_plot = common_files

    print(f"\nFound {len(files_to_plot)} v_snapshot files to plot; generating overlays...")

    bins = np.linspace(-60, -50, 41)
    colors = ['#3d3d3dff', '#2382f9fb', '#d53529ff']

    for basename in sorted(files_to_plot):
        fig, ax = plt.subplots(figsize=(1.4, 1.5))
        font_size = 6.64
        
        # Set transparent backgrounds
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        max_y = 0

        for idx, (ratio, d) in enumerate(selected_dirs):
            v_path = os.path.join(d, basename)
            v_snapshot = np.loadtxt(v_path)
            counts, _ = np.histogram(v_snapshot, bins=bins, density=False)
            max_y = max(max_y, counts.max() if counts.size else 0)

            # Fit Gaussian (no text on plot)
            v_mean = np.mean(v_snapshot)
            v_std = np.std(v_snapshot)
            gauss_r2 = np.nan
            gauss_params = None
            try:
                fit_params, _ = curve_fit(gaussian, bin_centers, counts,
                                          p0=[np.max(counts), v_mean, v_std])
                gauss_params = fit_params
                residuals = counts - gaussian(bin_centers, *fit_params)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((counts - np.mean(counts)) ** 2)
                gauss_r2 = 1 - (ss_res / ss_tot)
            except Exception:
                pass

            ax.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                    color=colors[idx], alpha=0.30)

            if gauss_params is not None:
                x_vals = np.linspace(bins[0], bins[-1], 200)
                ax.plot(x_vals, gaussian(x_vals, *gauss_params),
                         color=colors[idx], lw=2, label=f'{ratio}:1')

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        
        # ax.axvline(-50, color='black', lw=1.2, ls='-')
        ax.set_xlim([-56, -50])
        ax.set_xticks([-55, -50])
        ax.set_xticklabels(['-55 mV', '-50'], fontsize=font_size)
        # ax.set_xlabel(r'$v$ (mV)', fontsize=font_size)
        
        # Set bottom spine thickness
        ax.spines['bottom'].set_linewidth(.676)
        # Set x tick thickness
        ax.tick_params(axis='x', width=.676, length=4, direction='inout')

        
        ax.set_ylim([0, 175])
        ax.set_yticks([])  # Remove y-axis ticks
        
        # Add scale bar for y-axis (size 100)
        scale_bar_x = -55.8  # Position on x-axis
        scale_bar_y_start = 10  # Starting position on y-axis
        scale_bar_height = 100  # Height of scale bar
        ax.plot([scale_bar_x, scale_bar_x], 
                [scale_bar_y_start, scale_bar_y_start + scale_bar_height], 
                'k-', lw=0.676)
        ax.text(scale_bar_x - 0.2, scale_bar_y_start + scale_bar_height/2, 
                '100', fontsize=font_size, ha='right', va='center')
        
        # ax.legend(fontsize=font_size, frameon=False)
        plt.tight_layout()

        overlay_name = basename.replace('_value.txt', '')
        save_base = os.path.join(output_root, f'figS3D_s{scale}_STDP{stdp_time}s_{overlay_name}_r{selected_ratios[0]}_r{selected_ratios[1]}_r{selected_ratios[2]}')
        plt.savefig(save_base + '.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig(save_base + '.svg', bbox_inches='tight', transparent=True)
        plt.close()
        print(f"  Saved overlay for {basename} -> {save_base}.png")

    print("\nFinished generating overlays.")


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
