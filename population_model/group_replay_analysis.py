"""
Analysis and visualization tools for population model replay simulations.
"""

import numpy as np
from matplotlib import pyplot as plt, rc
import matplotlib as mpl
import pandas as pd
from population_model.theory import cond_fin_q_inf_t, cond_fin_q_fin_t
from scipy.stats import pearsonr
from general_code.aux_functions import xprint, bool_from_str, float_from_str

rc('mathtext', fontset='stix')
rc('font', family='sans-serif')
mpl.rcParams['hatch.linewidth'] = 2.0

def get_precision(number):
    """
    Determine the number of decimal places in a floating point number.
    
    Counts the minimum number of decimal places needed to represent the number
    without loss of precision.
    
    Args:
        number (float): The number to analyze.
    
    Returns:
        int: Number of decimal places (0 if integer).
    """
    n_dec = 0
    while 1:
        # Check if multiplying by 10^n_dec yields an integer
        remainder = number * (10 ** n_dec) % 10
        if remainder == 0:
            if n_dec > 0:
                n_dec -= 1
            break
        n_dec += 1

    return n_dec


def get_replay_pivot(group_path, n_asb):
    """
    Aggregate replay test results into a pivot table.
    
    Reads raw simulation results from a group directory, converts string data to
    appropriate types, filters to the last few assemblies, and creates a pivot table
    summarizing replay success, speed, and width for each (w_ff, w_rc) parameter pair.
    
    Args:
        group_path (str): Path to the directory containing simulation results.
        n_asb (int): Total number of assemblies in the simulation.
    
    Returns:
        None. Writes pivot table to '0_group_replay_pivot.txt' in group_path.
    """
    # Read test results file and create a DataFrame
    file_path = group_path + '/0_group_replay.txt'
    file = open(file_path, 'r')
    data = []
    for line in file.readlines():
        data.append(line.replace('\n', '').split(' \t '))
    file.close()

    df = pd.DataFrame(data=data[1:], columns=data[0])

    # Convert from string to appropriate numeric types
    df['w_ff'] = float_from_str(df['w_ff'])
    df['w_rc'] = float_from_str(df['w_rc'])

    df['replay'] = bool_from_str(df['replay'])
    for col_name in ['asb', 'std', 'cross-time']:
        df[col_name] = float_from_str(df[col_name])

    # Remove failed replays (set standard deviation to NaN)
    df.loc[df['replay'] == False, 'std'] = np.nan

    # Only keep statistics from last 3 assemblies (asymptotic propagation)
    df.loc[df['asb'] < n_asb - 2, 'std'] = np.nan

    # Create pivot table aggregating metrics for each (w_ff, w_rc) parameter pair
    replay_pivot = df.pivot_table(values=['replay', 'cross-time', 'std'],
                                  columns=['w_ff', 'w_rc'],
                                  aggfunc={'replay': np.mean,
                                           'cross-time': np.mean,
                                           'std': np.mean}).T

    replay_pivot.reset_index(inplace=True)

    print(replay_pivot.to_string(index=False))

    # Save pivot table to disk
    replay_pivot.to_csv(group_path + '/0_group_replay_pivot.txt', index=False)


def group_plot(group_path, field, n_asb=10,
               draw_levels=False, plot_theory=False, max_cbar=None):
    """
    Generate heatmap visualization of replay metrics across parameter space.
    
    Creates a 2D heatmap showing replay speed or width as a function of recurrent (w_rc)
    and feedforward (w_ff) connection weights. Optionally overlays theoretical boundaries.
    
    Args:
        group_path (str): Path to directory containing pivot table results.
        field (str): Metric to plot - 'asy_speed' for asymptotic propagation speed or 
                    'asy_width' for asymptotic temporal spread (FWHM).
        n_asb (int): Number of assemblies in simulation (default: 10).
        draw_levels (bool): Whether to draw contour levels (default: False).
        plot_theory (bool): Whether to overlay theoretical boundaries (default: False).
        max_cbar (float): Maximum colorbar value for plots (default: None).
    
    Returns:
        None. Saves heatmap as PNG in group_path.
    """
    # Configure field-specific visualization parameters
    if field == 'asy_speed':
        pivot_field = 'cross-time'      # Use assembly crossing time as raw metric
        color_map = 'gist_heat'         # Hot colors for faster speeds
        label_ = r'Speed (1 / time step)'
        max_speed_inv=2.5
        min_speed = 1 / max_speed_inv
        v_min = min_speed
        v_max = 1.0
    elif field == 'asy_width':
        pivot_field = 'std'             # Use standard deviation as raw metric
        color_map = 'gist_heat_r'       # Reversed heat (cold = narrow)
        label_ = r'FWHM (time step)'
        v_min=0.0
        v_max=5.0
    else:
        raise ValueError(f"Unknown field '{field}'. Choose from 'asy_speed', 'asy_width'.")

    # Read group results pivot table
    file_path = group_path + '/0_group_replay_pivot.txt'
    df = pd.read_csv(file_path)

    # Extract unique parameter values and sort them
    val_x = np.sort(np.unique(df['w_ff'].values))
    val_y = np.sort(np.unique(df['w_rc'].values))

    # Create 2D grid and initialize result matrices
    xx, yy = np.meshgrid(val_x, val_y, indexing='xy')
    zz = np.zeros_like(xx)          # Metric values (speed or width)
    zz_success = np.zeros_like(xx)  # Replay success rate

    # Populate grid with aggregated results
    for i in range(len(val_x)):
        for j in range(len(val_y)):
            # Get metric value for this parameter pair
            z = df.loc[(df['w_ff'] == val_x[i]) & (df['w_rc'] == val_y[j])][pivot_field].values
            if len(z) == 1:
                zz[j, i] = z[0]
            # Get success rate for this parameter pair
            z_success = df.loc[(df['w_ff'] == val_x[i]) & (df['w_rc'] == val_y[j])]['replay'].values
            if len(z_success) == 1:
                zz_success[j, i] = z_success[0]

    # Initialize figure
    fig_height = 1.8
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('darkgray')  # Gray background for NaN/missing values

    # Convert raw metrics to display units
    if field == 'asy_speed':        
        # Bin crossing times into discrete speed categories (n_asb/10 steps)
        zz_ceil = np.ceil(zz / (n_asb/10)) * (n_asb/10)
        # Convert crossing time to speed (assemblies per time step)
        zz_plot = n_asb / zz_ceil

    else: # field == 'asy_width'
        # Convert standard deviation to Full Width at Half Maximum (FWHM)
        zz_plot = zz * 2 * np.sqrt(2 * np.log(2))

    # Draw heatmap using pcolormesh (cell-centered values)
    dx = val_x[1] - val_x[0]
    x_coord = np.append(val_x, val_x[-1] + dx) - dx / 2
    dy = val_y[1] - val_y[0]
    y_coord = np.append(val_y, val_y[-1] + dy) - dy / 2
    color_plot = ax.pcolormesh(x_coord, y_coord, zz_plot,
                               vmin=v_min, vmax=v_max, cmap=color_map, 
                               rasterized=True)

    # Create fine-grained coordinate arrays for smooth theory line plotting
    step_size_x = 0.001
    step_size_y = 0.005
    range_x = np.arange(step_size_x, val_x[-1] + dx, step_size_x)
    range_y = np.arange(step_size_y, val_y[-1] + 2 * dy, step_size_y)

    # Overlay 45-degree line showing R + F = 1 boundary
    ax.plot(range_x, 1 - range_x, color='black', lw=1.0)
    ax.plot(range_x, 1 - range_x, color='darkgray', lw=0.75, alpha=1.0)

    # Configure colorbar with discrete speed bins or continuous width scale
    if field == 'asy_speed':
        if max_cbar is None:
            # Auto-scale: create discrete speed bins from max_speed_inv to near-instant
            speed_vals = np.array([n_asb/step for step in np.arange(n_asb*max_speed_inv, n_asb - 1, -n_asb/10)])
            boundaries = np.array([n_asb / step for step in
                                   np.arange(n_asb * max_speed_inv + 0.5 * (n_asb/10), n_asb - 1.5 * (n_asb/10), -n_asb/10)])
            cbar = fig.colorbar(color_plot, boundaries=boundaries, ax=ax, 
                                location='top', pad=0.02, extend='min')
        else:
            # Manual max: create bins up to specified maximum speed
            speed_vals = np.array([n_asb / step for step in np.arange(n_asb * max_cbar, n_asb - 1, -n_asb/10)])
            boundaries = np.array([n_asb / step for step in
                                   np.arange(n_asb * max_cbar + 0.5 * (n_asb/10), n_asb - 1.5 * (n_asb/10), -n_asb/10)])
            cbar = fig.colorbar(color_plot, boundaries=boundaries, ax=ax, 
                                location='top', pad=0.02)
        
        # Label every 5th speed value with fractional notation (10/n format)
        speed_val_ticks = speed_vals[::5]
        cbar.set_ticks(speed_val_ticks)
        tick_labels = ['10/%.0f' % (s_idx) for s_idx in (10 / speed_val_ticks)]
        cbar.set_ticklabels(tick_labels)
       
        # Mark reference speed with vertical line
        cbar.ax.axvline(x=1/max_speed_inv, color='lightgray', lw=1.0, ls='dotted')
    
    else: # field == 'asy_width'
        # Continuous colorbar for width with upper extension
        cbar = fig.colorbar(color_plot, ax=ax, location='top', pad=0.02, extend='max')
        # Mark reference widths with vertical lines
        for vline_ in [2.0, 4.0]:
            cbar.ax.axvline(x=vline_, color='lightgray', lw=1.0, ls='dotted')
        cbar.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Draw contour borders for discrete speed thresholds
    if draw_levels:
        if field == 'asy_speed':
            # Iterate through reference speeds and draw boundaries where speed changes
            for border_val in [10/25, 10/50, 10/100]:
                for i in range(zz_plot.shape[0] - 1, 0, -1):
                    for j in range(zz_plot.shape[1] - 1, 0, -1):
                        if zz_plot[i, j] >= border_val:
                            # Draw left edge where speed drops below threshold
                            if (zz_plot[i, j - 1] < border_val) or np.isnan(zz_plot[i, j - 1]):
                                ax.plot([x_coord[j], x_coord[j]], [y_coord[i], y_coord[i + 1]], 
                                        color='lightgray', linewidth=1.0, ls='dotted')
                            # Draw lower edge where speed drops below threshold
                            if (zz_plot[i - 1, j] < border_val) or np.isnan(zz_plot[i - 1, j]):
                                ax.plot([x_coord[j], x_coord[j + 1]], [y_coord[i], y_coord[i]], 
                                        color='lightgray', linewidth=1.0, ls='dotted')
    
        # Draw contour borders for discrete width thresholds
        else: # field == 'asy_width'
            # Iterate through reference widths and draw boundaries where width changes
            for border_val in [2, 4, 8, 16]:
                for i in range(zz_plot.shape[0] - 1, 0, -1):
                    for j in range(zz_plot.shape[1] - 1, 0, -1):
                        if zz_plot[i, j] < border_val:
                            # Draw right edge where width exceeds threshold
                            if j < zz_plot.shape[1] - 1:
                                if (zz_plot[i, j - 1] >= border_val) or np.isnan(zz_plot[i, j - 1]):
                                    ax.plot([x_coord[j], x_coord[j]], [y_coord[i], y_coord[i + 1]], 
                                            color='lightgray', linewidth=1.0, ls='dotted')
                            # Draw upper edge where width exceeds threshold
                            if i < zz_plot.shape[0] - 1:
                                if (zz_plot[i - 1, j] >= border_val) or np.isnan(zz_plot[i - 1, j]):
                                    ax.plot([x_coord[j], x_coord[j + 1]], [y_coord[i], y_coord[i]], 
                                            color='lightgray', linewidth=1.0, ls='dotted')
    

    # Overlay theoretical speed boundaries if requested
    if plot_theory and (field == 'asy_speed'):
        for speed_val in np.append(speed_vals, 0):
            # Compute theoretical boundary for this speed (0 = infinite time limit)
            if speed_val > 0:
                theor_check = cond_fin_q_fin_t(range_x, range_y, n_asb, int(n_asb / speed_val))
            else:
                theor_check = cond_fin_q_inf_t(range_x, range_y, n_asb)
            # Extract boundary coordinates from theoretical condition matrix
            theor_true = np.argwhere(theor_check==True)
            _, indices = np.unique(theor_true[:, 1], return_index=True)
            xxx = range_x[theor_true[indices][:, 0]]
            yyy = range_y[theor_true[indices][:, 1]]
            # Draw boundary with black outline and color-coded interior
            ax.plot(xxx, yyy, color='black', lw=1.5)
            if speed_val > 0:
                cmap = plt.get_cmap(color_map)
                line_color = cmap((speed_val - min_speed)/(v_max - min_speed))
                ax.plot(xxx, yyy, color=line_color, lw=1)

    # Format colorbar appearance
    cbar.ax.tick_params(length=3, width=0.5, labelsize=font_size, pad=0)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(label_, size=font_size, labelpad=5)

    # Format axes appearance
    ax.tick_params(length=3, width=0.5, direction='out', labelsize=font_size)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(0.5)
    ax.set_xlim(val_x[0], val_x[-1])
    ax.set_ylim(val_y[0], val_y[-1])
    ax.set_xlabel(r'Feedforward ($F$)', fontsize=font_size)
    ax.set_ylabel(r'Recurrent ($R$)', fontsize=font_size)

    # Save figure to disk
    plt.savefig(group_path + '/0_group_%s.png' % field, dpi=300, bbox_inches='tight')


def get_correlations(group_path, n_asb):
    """
    Analyze and visualize correlation between replay speed and temporal width.
    
    Computes Pearson correlation between inverse speed (time/assembly) and FWHM,
    fits a linear regression, and generates a scatter plot with the fitted line.
    Saves visualization to disk.
    
    Args:
        group_path (str): Path to directory containing pivot table results.
        n_asb (int): Number of assemblies in simulation.
    
    Returns:
        None. Saves correlation statistics to text file and scatter plot as PNG.
    """
    # Read aggregated replay results
    file_path = group_path + '/0_group_replay_pivot.txt'
    df = pd.read_csv(file_path)

    # Initialize figure
    fig_height = 1.18
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Extract and filter data
    q=n_asb
    # Convert crossing time to speed (assemblies per time step)
    speeds = np.array(q/df['cross-time'].values)
    speeds = speeds[~np.isnan(speeds)]
    # Extract standard deviations and filter NaNs
    widths_all = np.array(df['std'].values)
    widths_all = widths_all[~np.isnan(widths_all)]

    # Convert standard deviation to Full Width at Half Maximum (FWHM)
    widths_all *= 2*np.sqrt(2*np.log(2))

    # Limit analysis to narrow replays (FWHM <= 8 time steps)
    widths = widths_all[widths_all <= 8]
    speeds = speeds[widths_all <= 8]

    # Compute Pearson correlation between inverse speed and width
    corrfile = group_path + '/0_group_corr.txt'
    c, p = pearsonr(1/speeds, widths)
    xprint('Correlation: c = %.3f, p = %f' % (c, p), corrfile)

    # Fit linear regression: FWHM = slope * (1/speed) + intercept
    coefficients = np.polyfit(1/speeds, widths, 1)
    slope, intercept = coefficients

    # Generate fitted line predictions
    widths_fit = slope * (1/speeds) + intercept

    # Calculate R²
    ss_res = np.sum((widths - widths_fit) ** 2)  # Residual sum of squares
    ss_tot = np.sum((widths - np.mean(widths)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    # Log regression statistics
    xprint('R2: %f' % r_squared, corrfile)
    xprint('slope: %f' % slope, corrfile)
    xprint('intercept: %s' % intercept, corrfile)

    # Plot scatter and regression line
    x_array = np.arange(0, np.max(1/speeds), 0.1)
    ax.scatter(1/speeds, widths, s=1, c='k')
    ax.plot(x_array, slope * x_array + intercept, ls='--', lw=2, c='gray')

    # Format axes
    ax.tick_params(length=3, width=0.5, direction='out', labelsize=font_size)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
    for spines in ['top', 'right']:
        ax.spines[spines].set_visible(False)

    ax.set_xlabel(r'Speed$^{-1}$ (time step)', fontsize=font_size)
    ax.set_ylabel('FWHM (time step)', fontsize=font_size)
    ax.set_ylim([0, 8])
    ax.set_yticks([0, 4, 8])
    ax.set_yticklabels(['0', '4', '8'], fontsize=font_size)
    ax.set_xlim([0, 11])
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '5', '10'], fontsize=font_size)

    # Save scatter plot to disk
    plt.savefig(group_path + '/0_group_corr.png', dpi=600, bbox_inches='tight')
