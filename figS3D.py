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
from spiking_networks.model2 import run_simulation


ei_ratio = 8  # ratio of excitatory to inhibitory neurons
scale = 'g_sqrt'   # scaling method for inhibitory synapses
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
    base_pattern = f'figS3D_r*_s{scale}_STDP{stdp_time}s_*'
    candidate_dirs = sorted(glob.glob(os.path.join(output_root, base_pattern)))

    if not candidate_dirs:
        print(f"No output directories matching {base_pattern} found under outputs/")
        return

    # Map ei_ratio -> list of dirs, pick latest per ratio
    ratio_to_dir = {}
    for d in candidate_dirs:
        try:
            name_part = os.path.basename(d)
            ratio_str = name_part.split('_')[1]  # r{ei}
            if ratio_str.startswith('r'):
                ei_val = int(ratio_str[1:])
            else:
                continue
            if ei_val not in ratio_to_dir:
                ratio_to_dir[ei_val] = []
            ratio_to_dir[ei_val].append(d)
        except Exception:
            continue

    print("Available ei_ratio values for this scale/stdp_time:")
    for r in sorted(ratio_to_dir.keys()):
        print(f"  r{r}: {len(ratio_to_dir[r])} runs")

    user_input = input("Enter three ei_ratio values (comma-separated, e.g., 2,4,8): ").strip()
    if not user_input:
        print("No ei_ratio values provided; aborting plot.")
        return

    try:
        selected_ratios = [int(v) for v in user_input.split(',') if v.strip()]
    except ValueError:
        print("Invalid input; please enter integers like 2,4,8")
        return

    if len(selected_ratios) != 3:
        print("Please provide exactly three ei_ratio values.")
        return

    selected_dirs = []
    for ratio in selected_ratios:
        if ratio not in ratio_to_dir:
            print(f"No runs found for ei_ratio={ratio} with scale={scale} and stdp_time={stdp_time}s")
            return
        dir_list = sorted(ratio_to_dir[ratio], key=os.path.getmtime, reverse=True)
        if len(dir_list) == 1:
            chosen_dir = dir_list[0]
        else:
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

    print(f"\nFound {len(common_files)} shared v_snapshot files; generating overlays...")

    bins = np.linspace(-60, -50, 41)
    colors = ['#3d3d3dff', '#2382f9fb', '#d53529ff']

    for basename in sorted(common_files):
        plt.figure(figsize=(3.0, 2.4))
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

            plt.bar(bins[:-1], counts, width=np.diff(bins), align='edge',
                    color=colors[idx], alpha=0.30, label=f'r{ratio} hist')

            if gauss_params is not None:
                x_vals = np.linspace(bins[0], bins[-1], 200)
                plt.plot(x_vals, gaussian(x_vals, *gauss_params),
                         color=colors[idx], lw=2, label=f'r{ratio} fit (R²={gauss_r2:.2f})')

        plt.axvline(-50, color='black', lw=1.2, ls='--')
        plt.xlim([-61, -49.5])
        plt.ylim([0, max_y * 1.15 if max_y > 0 else 1])
        plt.xticks([-60, -55, -50], ['-60', '', '-50'])
        plt.xlabel(r'$v$ (mV)', fontsize=9)
        plt.ylabel('count', fontsize=9)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()

        overlay_name = basename.replace('_value.txt', '')
        save_base = os.path.join(output_root, f'figS3D_s{scale}_STDP{stdp_time}s_{overlay_name}_r{selected_ratios[0]}_r{selected_ratios[1]}_r{selected_ratios[2]}')
        plt.savefig(save_base + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_base + '.svg', bbox_inches='tight')
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
