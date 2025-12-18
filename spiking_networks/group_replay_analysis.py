"""
group_replay_analysis.py
Functions for analyzing and visualizing group statistics from replay tests.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, rc, patches
from matplotlib.colors import LogNorm
import pandas as pd
from population_model.theory import cond_inf_q
from scipy.stats import pearsonr
from general_code.aux_functions import xprint, bool_from_str, float_from_str

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']


def get_replay_pivot(group_path, qual_thres=0.8, act_low_thres=0.9, act_up_thres=1.1):
    """
    Parse replay test results and generate a pivot table summarizing group statistics.

    Args:
        group_path (str): Path to the group results folder.
        qual_thres (float): Minimum replay quality threshold.
        act_low_thres (float): Lower activity threshold for replay success.
        act_up_thres (float): Upper activity threshold for replay success.

    Returns:
        pd.DataFrame: Pivoted group statistics.
    """
    # Read test results and create DataFrame
    file_path = group_path + '/0_group_replay.txt'
    file = open(file_path, 'r')
    data = []
    for line in file.readlines():
        data.append(line.replace('\n', '').split(' \t '))
    file.close()
    df = pd.DataFrame(data=data[1:], columns=data[0])

    # Convert columns from string to appropriate types
    df['p_ff'] = float_from_str(df['p_ff'])
    df['p_rc'] = float_from_str(df['p_rc'])

    df['replay'] = bool_from_str(df['replay'])
    df['explosion'] = bool_from_str(df['explosion'])

    for col_name in ['asy_speed', 'asy_act', 'asy_width']:
        df[col_name] = float_from_str(df[col_name])

    # Invert speed values for analysis
    df['asy_speed'] = 1 / df['asy_speed']

    # Set replay status to False if activity is outside desired thresholds
    df.loc[df['asy_act'] <= act_low_thres, 'replay'] = False
    df.loc[df['asy_act'] >= act_up_thres, 'replay'] = False

    # Mark explosion as True if activity exceeds upper threshold
    df.loc[df['asy_act'] >= act_up_thres, 'explosion'] = True

    # Replace failed replays with NaN so they don't affect mean calculations
    for col_name in ['asy_speed', 'asy_act', 'asy_width']:
        if col_name in df:
            df.loc[df['replay'] == False, col_name] = np.nan

    # Custom explosion aggregation logic:
    # Explosion is True if: 
    #    (at most 1 failed replay did NOT have explosion AND there is at least one explosion)
    #    OR (at least 40% of all trials had explosion).
    # Note: Replay may have failed due to coincidental spontaneous activity, not just explosion.
    def explosion_logic(subdf):
        
        failed = subdf[subdf['replay'] == False]
        if len(failed) == 0:
            return False
        
        any_explosion = np.sum(failed['explosion']) > 0
        non_expl_failures = np.sum(~failed['explosion'])
        check1 = (non_expl_failures <= 1) and any_explosion

        check2 = np.mean(subdf['explosion']) >= 0.4
        
        return bool(check1 or check2)
    
    explosion_status = df.groupby(['p_ff', 'p_rc']).apply(explosion_logic).reset_index()
    explosion_status.columns = ['p_ff', 'p_rc', 'explosion']
    df = pd.merge(df.drop(columns=['explosion']), explosion_status, on=['p_ff', 'p_rc'], how='left')

    # Create pivot table summarizing statistics for each (p_ff, p_rc) pair
    replay_pivot = df.pivot_table(values=['replay', 'explosion', 'asy_speed', 'asy_act', 'asy_width'],
                                    columns=['p_ff', 'p_rc'],
                                    aggfunc={'replay': [np.sum, 'size'],
                                            'explosion': np.max,
                                            'asy_speed': np.mean,
                                            'asy_act': np.mean,
                                            'asy_width': np.mean,}).T

    # Check that all tests are complete
    counts_array = replay_pivot['replay']['size'].values
    max_trials = np.max(counts_array)
    equal_ = np.all(counts_array == max_trials)
    if not equal_:
        print('ERROR Not all tests are complete!')
        replay_array = replay_pivot['sum'].values
        if np.all(replay_array[counts_array != max_trials] == 0):
            print('All incomplete tests have quality 0! printing...')
        else:
            exit()

    # Calculate replay quality (percentage of successful trials)
    replay_pivot['quality'] = replay_pivot['replay']['sum'] / max_trials
    
    replay_pivot.drop(labels='replay', axis=1, inplace=True)
    replay_pivot.reset_index(inplace=True)

    # Remove statistics if quality is below threshold
    for field_ in ['asy_speed', 'asy_width', 'asy_act']:
         replay_pivot.loc[replay_pivot['quality'] < qual_thres, field_] = np.nan

    # Flatten multi-level column headers
    flat_cols = []
    for i in replay_pivot.columns:
        flat_cols.append(i[0])
    replay_pivot.columns = flat_cols

    print(replay_pivot.to_string(index=False))
    replay_pivot.to_csv(group_path + '/0_group_replay_pivot.txt', index=False)

    return replay_pivot



def group_plot(group_path, field,
               label_='',
               cond1=None, cond2=None,
               cond3=[], time_step=None, dist_width=None,
               cbar_ticks_=[], cbar_ticklabels_=[], vspan_=[],
               square_coords=[],
               color_map='gist_heat'):

    # Read group results pivot table
    replay_file_path = group_path + '/0_group_replay_pivot.txt'
    df_replay = pd.read_csv(replay_file_path)

    # Convert probabilities to percentages
    df_replay['p_ff'] = df_replay['p_ff'] * 100
    df_replay['p_rc'] = df_replay['p_rc'] * 100

    # Sort values for plotting
    val_x = np.sort(np.unique(df_replay['p_ff'].values))
    val_y = np.sort(np.unique(df_replay['p_rc'].values))

    # Get coordinates for pcolormesh
    dx = val_x[1] - val_x[0]
    x_coord = np.append(val_x, val_x[-1] + dx) - dx / 2

    dy = val_y[1] - val_y[0]
    y_coord = np.append(val_y, val_y[-1] + dy) - dy / 2

    # Create figure and axes
    fig_height = 1.18
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Iterate over each parameter pair
    xx, yy = np.meshgrid(val_x, val_y, indexing='xy')
    zz = np.zeros_like(xx)
    for i in range(len(val_x)):
        for j in range(len(val_y)):

            # Mark explosions with hatching
            z_explosion = df_replay.loc[(df_replay['p_ff'] == val_x[i]) & (df_replay['p_rc'] == val_y[j])]['explosion'].values
            if z_explosion[0] > 0:
                rect = patches.Rectangle(
                    (x_coord[i], y_coord[j]), dx, dy,
                    linewidth=0, edgecolor=None, facecolor='none', 
                    hatch='////////', alpha=0.5
                )
                ax.add_patch(rect)

            # Get field value for parameter pair
            z = df_replay.loc[(df_replay['p_ff'] == val_x[i]) & (df_replay['p_rc'] == val_y[j])][field].values
            if len(z) == 1:
                zz[j, i] = z[0]

    # Prepare color mesh for plotting
    zz_plot = zz.copy()

    # Define color scale limits
    if len(vspan_) > 0:
        vmin_ = vspan_[0]
        vmax_ = vspan_[1]
    else:
        vmin_ = np.nanmin(zz)
        vmax_ = np.nanmax(zz)

    # Plot condition 1 (if provided)
    if cond1 is not None:
        range_x = np.arange(0, cond1 + dx , dx)
        ax.plot(range_x, (cond1 - range_x), color='white', lw=1.0, ls='--')

    # Plot condition 2 (if provided)
    if cond2 is not None:
        ax.axvline(cond2, color='white', lw=1.0, ls='--')

    def get_speed_color(speed_):
        # Get color corresponding to a given speed
        if speed_ == 0:
            # if given speed is 0, find actual minimum speed in data
            real_speed = np.nanmin(zz)
            color_ = 'black'
        else:
            real_speed = speed_
            color_norm = mpl.colors.Normalize(vmin=vmin_, vmax=vmax_)
            cmap = mpl.cm.get_cmap(color_map)
            color_ = cmap(color_norm(speed_))

        return real_speed, color_
    
    # Plot condition 3 borders if speeds are provided
    if len(cond3) > 0:

        # Iterate through each condition-3 speed
        for speed_ in np.sort(cond3):

            if speed_ > np.nanmax(zz):
                break

            real_speed, color_ = get_speed_color(speed_)

            # Convert spiking network speed to population model speed
            s0 = np.min([real_speed * time_step, 1])

            # Find border between success and failure for condition 3 in x-y plane
            range_x = np.arange(0.01, val_x[-1] + dx, 0.01) / dist_width
            range_y = np.arange(0.05, val_y[-1] + 2 * dy, 0.05) / dist_width
            theor_check = cond_inf_q(range_x, range_y, s0)
            theor_true = np.argwhere(theor_check == True)
            _, indices = np.unique(theor_true[:, 1], return_index=True)
            xxx = range_x[theor_true[indices][:, 0]] * dist_width
            yyy = range_y[theor_true[indices][:, 1]] * dist_width

            # Plot the border
            ax.plot(xxx, yyy, color='black', lw=1.5, ls='-')
            ax.plot(xxx, yyy, color=color_, lw=1, ls='-')
    
            # Assign same color to all speeds larger than given speed
            zz_plot[zz >= speed_] = speed_

    # Create color plot and color bar
    color_plot = ax.pcolormesh(x_coord, y_coord, zz_plot,
                               vmin=vmin_, vmax=vmax_,
                               cmap=color_map, rasterized=True)
    ax.set_facecolor('darkgray')
    cbar = fig.colorbar(color_plot, ax=ax, location='top', pad=0.02)

    # Add color bar patches for each condition-3 speed
    for k in range(len(cond3)):
        real_speed, color_ = get_speed_color(cond3[k])

        if k < len(cond3) - 1:
            cbar.ax.add_patch(patches.Rectangle(
                (real_speed, 0),
                cond3[k+1] - real_speed, 1,
                lw=0, color=color_, alpha=1.0))
        else:
            max_speed_ = np.nanmax(zz)
            cbar.ax.add_patch(patches.Rectangle(
                (real_speed, 0),
                max_speed_ - real_speed, 1,
                lw=0, color=color_, alpha=1.0))
        
    # Grey out regions outside min-max range
    if len(vspan_) > 0:
        if np.nanmax(zz) > np.nanmin(zz):
            cbar.ax.add_patch(patches.Rectangle(
                (vspan_[0], 0),
                np.nanmin(zz) - vspan_[0], 1,
                lw=0, color='darkgray', alpha=0.8))
            cbar.ax.add_patch(patches.Rectangle(
                (np.nanmax(zz), 0),
                vspan_[1] - np.nanmax(zz), 1,
                lw=0, color='darkgray', alpha=0.8))
        else:
            vspan_length = vspan_[1] - vspan_[0]
            cbar.ax.add_patch(patches.Rectangle(
                (vspan_[0], 0),
                0.99*(np.nanmin(zz) - vspan_[0]), 1,
                lw=0, color='darkgray', alpha=0.8))
            cbar.ax.add_patch(patches.Rectangle(
                (np.nanmax(zz)*1.01, 0),
                vspan_[1] - np.nanmax(zz), 1,
                lw=0, color='darkgray', alpha=0.8))

    # Add lines and squares for specified coordinates
    for (x_, y_, color_) in square_coords:
        ax.add_patch(patches.Rectangle((x_ - dx / 2, y_ - dy / 2), dx, dy,
                                edgecolor=color_, fill=None, lw=0.5))
        ax.plot([x_ - 0.9 * dx / 2, x_ + 0.9 * dx / 2], [y_ - 0.9 * dy / 2, y_ + 0.9 * dy / 2], color=color_, lw=0.5)
        ax.plot([x_ - 0.9 * dx / 2, x_ + 0.9 * dx / 2], [y_ + 0.9 * dy / 2, y_ - 0.9 * dy / 2], color=color_, lw=0.5)
        line_ = df_replay.loc[np.isclose(df_replay['p_ff'], x_) & np.isclose(df_replay['p_rc'], y_)][field].values[0]
        cbar.ax.axvline(line_, c='k', lw=2)
        cbar.ax.axvline(line_, c=color_, lw=1)

    # Color bar settings
    if len(cbar_ticks_) > 0:
        cbar.set_ticks(cbar_ticks_)
        if len(cbar_ticklabels_) > 0:
            cbar.set_ticklabels(cbar_ticklabels_)
        else:
            cbar.set_ticklabels(cbar_ticks_)
    cbar.ax.tick_params(length=1.5, width=0.5, labelsize=font_size, pad=0)
    cbar.outline.set_linewidth(0.5)

    # Axes settings
    if (len(cond3) > 0) or (cond2 is not None) or (cond1 is not None):
        ax.set_xlim(val_x[0], val_x[-1])
        ax.set_ylim(val_y[0], val_y[-1])    
    else:
        ax.set_xlim(val_x[0]-dx/2, val_x[-1]+dx/2)
        ax.set_ylim(val_y[0]-dy/2, val_y[-1]+dy/2)
    ax.tick_params(length=1.5, width=0.5, direction='out', labelsize=font_size)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(0.5)
    ax.set_xticks(val_x)
    x_labels = []
    x_ticks = []
    for i in np.arange(0, len(val_x), 4):
        x_ticks += [val_x[i]]
        x_labels += [f'{int(val_x[i]):,}']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels=x_labels, fontsize=font_size)
    ax.set_yticks(val_y)
    y_labels = ['0']
    y_ticks = [0]
    for i in np.arange(4, len(val_y), 4):
        y_ticks += [val_y[i]]
        y_labels += [f'{int(val_y[i]):,}']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels=y_labels, fontsize=font_size)

    # Axis labels and title
    ax.set_xlabel(r'$p_f$ (%)', fontsize=font_size)
    ax.set_ylabel(r'$p_r$ (%)', fontsize=font_size)
    ax.set_title(label_, fontsize=font_size, pad=20)

    # Save figure to disk
    figname = group_path + '/0_group_%s' % field
    plt.savefig(figname + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(figname + '.svg', dpi=600, bbox_inches='tight')

    plt.close()
    

def get_correlations(group_path):
    """
    Compute and plot the correlation between replay speed and width for a group test.
    Saves correlation statistics and a scatter plot with fitted line.

    Args:
        group_path (str): Path to the test group folder.

    Returns:
        None. Results are saved to disk.
    """

    # Read group results pivot table
    replay_file_path = group_path + '/0_group_replay_pivot.txt'
    df_replay = pd.read_csv(replay_file_path)


    # Plot results
    fig_height = 1.18
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    speeds = np.array(df_replay['asy_speed'].values)
    speeds = speeds[~np.isnan(speeds)]
    widths = np.array(df_replay['asy_width'].values)
    widths = widths[~np.isnan(widths)]

    corrfile = group_path + '/0_group_corr.txt'
    c, p = pearsonr(1/speeds, widths)
    xprint('Correlation: c = %.3f, p = %f' % (c, p), corrfile)

    # Fit a line to the data
    coefficients = np.polyfit(1/speeds, widths, 1)
    slope, intercept = coefficients

    # Generate fitted line values
    widths_fit = slope * (1/speeds) + intercept

    # Calculate R-squared value
    ss_res = np.sum((widths - widths_fit) ** 2)
    ss_tot = np.sum((widths - np.mean(widths)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    xprint('R2: %f' % r_squared, corrfile)
    xprint('slope: %f' % slope, corrfile)
    xprint('intercept: %s' % intercept, corrfile)

    x_array = np.arange(0, 10.5, 0.1)
    ax.scatter(1/speeds, widths, s=1, c='k')
    ax.plot(x_array, slope * x_array + intercept, ls='--', lw=2, c='gray')

    # Edit axes appearance
    ax.tick_params(length=3, width=0.5, direction='out', labelsize=font_size)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
    for spines in ['top', 'right']:
        ax.spines[spines].set_visible(False)

    ax.set_xlabel(r'Speed$^{-1}$ (ms)', fontsize=font_size)
    ax.set_ylabel(r'FWHM (ms)', fontsize=font_size)
    ax.set_ylim([4, 10.5])
    ax.set_xlim([0, 10.5])
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '5', '10'], fontsize=font_size)
    ax.set_yticks([4, 7, 10])
    ax.set_yticklabels(['4', '7', '10'], fontsize=font_size)

    # Save results figure to disk
    plt.savefig(group_path + '/0_group_corr.png', dpi=600, bbox_inches='tight')
    plt.close()
