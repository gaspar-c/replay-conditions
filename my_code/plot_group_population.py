import os
import numpy as np
import math
from matplotlib import pyplot as plt, rc, colors as mcolors
import matplotlib as mpl
from matplotlib.colors import LogNorm, PowerNorm, Normalize, BoundaryNorm
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import scipy.optimize
import pandas as pd
from my_code.discrete_model import simplified_theory
from scipy.stats import pearsonr
from my_code.aux_functions import xprint

rc('text', usetex=True)
mpl.rcParams['hatch.linewidth'] = 2.0

def get_precision(number):
    n_dec = 0
    while 1:
        remainder = number * (10 ** n_dec) % 10
        if remainder == 0:
            if n_dec > 0:
                n_dec -= 1
            break
        n_dec += 1

    return n_dec


def bool_from_str(str_array):
    """
    convert string array to boolean array
    """

    bool_array = np.zeros_like(str_array, dtype=bool)
    for k in range(len(str_array)):
        if str_array[k] == 'True':
            bool_array[k] = True
        elif str_array[k] == 'False':
            bool_array[k] = False
        else:
            print('ERROR str not True/False!')
            exit()

    return bool_array



def float_from_str(str_array, unit=1):
    """
    convert string array to value array
    """
    np_str = np.array(str_array, dtype=str)
    star_position = np.char.find(np_str, '*')
    val_array = np.zeros_like(np_str, dtype=float) * unit

    for k in range(len(str_array)):
        if star_position[k] > 0:
            val_array[k] = float(np_str[k][:star_position[k]])
        else:
            val_array[k] = float(np_str[k])

    return val_array


def get_pivot(group_path, var1, var2, n_asb=None):
    """
    """
    # read test results file and create a DataFrame
    file_path = group_path + '/0_group_square.txt'
    file = open(file_path, 'r')
    data = []
    for line in file.readlines():
        data.append(line.replace('\n', '').split(' \t '))
    file.close()

    df = pd.DataFrame(data=data[1:], columns=data[0])

    # convert from string to values
    df[var1] = float_from_str(df[var1])
    df[var2] = float_from_str(df[var2])

    df['replay'] = bool_from_str(df['replay'])
    for col_name in ['asb', 'cross', 'mean', 'std', 'cross-time']:
        df[col_name] = float_from_str(df[col_name])

    # remove failed replays
    df.loc[df['replay'] == False, 'std'] = np.nan

    # only keep last 3
    if n_asb is not None:
        df.loc[df['asb'] < n_asb - 2, 'std'] = np.nan

    # create pivot table that counts successful replays for each (var1, var2) pair
    replay_pivot = df.pivot_table(values=['replay', 'cross-time', 'std'],
                                  columns=[var1, var2],
                                  aggfunc={'replay': np.mean,
                                           'cross-time': np.mean,
                                           'std': np.mean}).T

    replay_pivot.reset_index(inplace=True)

    print(replay_pivot.to_string(index=False))

    replay_pivot.to_csv(group_path + '/0_group_square_pivot.txt', index=False)


def group_plot(group_path, field,
               name_x, name_y, xlabel_, ylabel_,
               n_asb=10, max_speed_inv=1.5, max_std=6,
               title_='',
               xlabel_dec=0, ylabel_dec=0,
               scale_x=1, scale_y=1, scale_z=1, skip_xlabel=2, skip_ylabel=2,
               xlabel_append='', fit_inv=False, plot_theory=False, max_cbar=None,
               pivot_file = '/0_group_square_pivot.txt'):
    """
    plot results of a group test (for panels C, D, E)

    Args:
        group_path: folder path of test group
        name_x: name of variable to plot on the x-axis
        name_y: name of variable to plot on the y-axis
        xlabel_: x-axis label
        ylabel_: y-axis label
        title_: plot title
        scale_x: scale of x-axis variable
        scale_y: scale of y-axis variable
        skip_xlabel: skip labels in x-axis
        skip_ylabel: skip labels in y-axis
        xlabel_append: append str to x label
        fit_inv: fit x.y=cte line to the test results
        plot_theory: plot theoretical results

    """
    file_path = group_path + pivot_file
    if field == 'cross-time':
        label_ = r'Speed (1 / time step)'
    elif field == 'std':
        label_ = r'FWHM (time step)'
    elif field == 'r0' or field == 'cross':
        field = 'cross'
        label_ = r'First assembly'
    elif field == 'replay':
        label_ = r'Success'

    # read group results pivot
    df = pd.read_csv(file_path)

    # scale values
    df[name_x] = df[name_x] * scale_x
    df[name_y] = df[name_y] * scale_y

    # sort values
    val_x = np.sort(np.unique(df[name_x].values))
    val_y = np.sort(np.unique(df[name_y].values))

    # get percentage of successful replays for each (x,y) value pair
    xx, yy = np.meshgrid(val_x, val_y, indexing='xy')
    zz = np.zeros_like(xx)
    zz_success = np.zeros_like(xx)

    for i in range(len(val_x)):
        for j in range(len(val_y)):
            z = df.loc[(df[name_x] == val_x[i]) & (df[name_y] == val_y[j])][field].values
            if len(z) == 1:
                zz[j, i] = z[0]
            z_success = df.loc[(df[name_x] == val_x[i]) & (df[name_y] == val_y[j])]['replay'].values
            if len(z_success) == 1:
                zz_success[j, i] = z_success[0]

    dx = val_x[1] - val_x[0]
    x_coord = np.append(val_x, val_x[-1] + dx) - dx / 2

    dy = val_y[1] - val_y[0]
    y_coord = np.append(val_y, val_y[-1] + dy) - dy / 2

    # plot results
    fig_height = 1.78
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_facecolor('darkgray')

    # color_map = get_custom_map()
    # color_map = 'gist_heat_r'
    color_map = 'gist_heat'
    if field == 'cross-time':
        min_speed = 1 / max_speed_inv
        if max_cbar is None:
            speed_vals = np.array([n_asb/step for step in np.arange(n_asb*max_speed_inv, n_asb - 1, -n_asb/10)])
        else:
            speed_vals = np.array([n_asb / step for step in np.arange(n_asb * max_cbar, n_asb - 1, -n_asb/10)])

        cmap = plt.get_cmap(color_map)
        v_max = 1.0
        # norm = PowerNorm(gamma=1.2, vmax=1.5, vmin=min_speed)  # max_speed_inv)
        norm = None
        if max_cbar is None:
            boundaries = np.array([n_asb / step for step in
                                   np.arange(n_asb * max_speed_inv + 0.5 * (n_asb/10), n_asb - 1.5 * (n_asb/10), -n_asb/10)])
        else:
            boundaries = np.array([n_asb / step for step in
                                   np.arange(n_asb * max_cbar + 0.5 * (n_asb/10), n_asb - 1.5 * (n_asb/10), -n_asb/10)])

        # group z values by s integers
        # zz_plot = np.ceil(10 * zz * scale_z / n_asb) / 10
        zz_ceil = np.ceil(zz / (n_asb/10)) * (n_asb/10)
        zz_plot = n_asb / zz_ceil

        color_plot = ax.pcolormesh(x_coord, y_coord, zz_plot,
                                   # norm=norm,
                                   vmin=min_speed,
                                   vmax=v_max,
                                   cmap=color_map, rasterized=True)

        # Create the colorbar from the invisible image
        # cbar = fig.colorbar(color_plot, ax=ax)
        if max_cbar is None:
            cbar = fig.colorbar(color_plot, boundaries=boundaries, ax=ax, location='top', pad=0.02,
                                extend='min')
        else:
            cbar = fig.colorbar(color_plot, boundaries=boundaries, ax=ax, location='top', pad=0.02)
        speed_val_ticks = speed_vals[::5]
        # speed_val_ticks = speed_vals[1::2]
        cbar.set_ticks(speed_val_ticks)
        tick_labels = ['10/%.0f' % (s_idx) for s_idx in (10 / speed_val_ticks)]
        cbar.set_ticklabels(tick_labels)
        cbar.ax.axvline(x=10/25.5, color='lightgray', lw=1.0, ls='dotted')
        # cbar.set_ticks([1.0, 2.0, 5.0, 10.0])
        # cbar.set_ticklabels(['1.0', '0.5', '0.2', '0.1'])

        # Draw the lower and left edges of the border squares for specific values where zz_plot values decrease
        specific_values = [10/25, 10/50, 10/100]
        colors = ['lightgray', 'lightgray', 'lightgray']
        for specific_val, color in zip(specific_values, colors):
            for i in range(zz_plot.shape[0] - 1, 0, -1):
                for j in range(zz_plot.shape[1] - 1, 0, -1):
                    if zz_plot[i, j] >= specific_val:
                        # Draw left edge
                        if (zz_plot[i, j - 1] < specific_val) or np.isnan(zz_plot[i, j - 1]):
                            ax.plot([x_coord[j], x_coord[j]], [y_coord[i], y_coord[i + 1]], color=color, linewidth=1.0, ls='dotted')
                        # Draw lower edge
                        if (zz_plot[i - 1, j] < specific_val) or np.isnan(zz_plot[i - 1, j]):
                            ax.plot([x_coord[j], x_coord[j + 1]], [y_coord[i], y_coord[i]], color=color, linewidth=1.0, ls='dotted')
    elif (field == 'std'):
        # color_map = get_custom_map(rev=True)
        color_map = 'gist_heat_r'

        cmap = plt.get_cmap(color_map)
        norm = None

        # convert SD to FWHM
        zz_plot = zz * 2 * np.sqrt(2 * np.log(2))

        color_plot = ax.pcolormesh(x_coord, y_coord, zz_plot,
                                   vmin=0.0,
                                   vmax=max_std,
                                   cmap=color_map, rasterized=True)

        # Create the colorbar from the invisible image
        cbar = fig.colorbar(color_plot, ax=ax, location='top', pad=0.02, extend='max')
        for vline_ in [2.0, 4.0]:
            cbar.ax.axvline(x=vline_, color='lightgray', lw=1.0, ls='dotted')

        cbar.set_ticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        # cbar.set_ticklabels(['0', '', '2', '', '4', ''])

        # Draw the lower and left edges of the border squares for specific values where zz_plot values decrease
        specific_values = [2, 4, 8, 16]
        # colors = ['white', 'white', 'white', 'white', 'white']
        colors = ['lightgray', 'lightgray', 'lightgray', 'lightgray', 'lightgray']
        for specific_val, color in zip(specific_values, colors):
            for i in range(zz_plot.shape[0] - 1, 0, -1):
                for j in range(zz_plot.shape[1] - 1, 0, -1):
                    if zz_plot[i, j] < specific_val:
                        # Draw right edge
                        if j < zz_plot.shape[1] - 1:
                            if (zz_plot[i, j - 1] >= specific_val) or np.isnan(zz_plot[i, j - 1]):
                                ax.plot([x_coord[j], x_coord[j]], [y_coord[i], y_coord[i + 1]], color=color, linewidth=1.0, ls='dotted')
                        # Draw upper edge
                        if i < zz_plot.shape[0] - 1:
                            if (zz_plot[i - 1, j] >= specific_val) or np.isnan(zz_plot[i - 1, j]):
                                ax.plot([x_coord[j], x_coord[j + 1]], [y_coord[i], y_coord[i]], color=color, linewidth=1.0, ls='dotted')
    elif field == 'cross':
        boundaries = np.array([asb_idx - 0.5 for asb_idx in np.arange(1, n_asb + 2)])
        color_map = 'gist_heat_r'
        color_plot = ax.pcolormesh(x_coord, y_coord, zz,
                                   cmap=color_map,
                                   vmin=1,
                                   vmax=n_asb,
                                   rasterized=True)
        cbar = fig.colorbar(color_plot, ax=ax, boundaries=boundaries, location='top', pad=0.02)
        # bar_ticks = cbar.ax.get_xticks()
        cbar.set_ticks(np.arange(1, n_asb + 1, 2))
        cbar.set_ticklabels(np.arange(1, n_asb + 1, 2))

    else:
        color_plot = ax.pcolormesh(x_coord, y_coord, zz,
                                   cmap=color_map, rasterized=True)
        cbar = fig.colorbar(color_plot, ax=ax, location='top', pad=0.02)

    ax.set_xlim(val_x[0], val_x[-1])
    ax.set_ylim(val_y[0], val_y[-1])

    # store axis limitsbound
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    # Define the step size for the x and y values
    step_size_x = 0.001
    step_size_y = 0.005

    # Create a new range of x and y values with the defined step size
    range_x = np.arange(step_size_x, val_x[-1] + dx, step_size_x)
    range_y = np.arange(step_size_y, val_y[-1] + 2 * dy, step_size_y)

    if ((field == 'replay') or (field == 'cross-time')) and plot_theory:
        for speed_val in np.append(speed_vals, 0):
            if speed_val > 0:
                t_step = int(n_asb / speed_val)
            else:
                t_step = np.nan
            theor_check = simplified_theory(range_x, range_y, n_asb, t_step, init_j=1.0)
            theor_true = np.argwhere(theor_check==True)
            unique, indices = np.unique(theor_true[:, 1], return_index=True)
            xxx = range_x[theor_true[indices][:, 0]]
            yyy = range_y[theor_true[indices][:, 1]]
            if speed_val > 0:
                ax.plot(xxx, yyy, color='black', lw=3)
                if norm is not None:
                    line_color = cmap(norm(speed_val))
                else:
                    line_color = cmap((speed_val - min_speed)/(v_max - min_speed))
                ax.plot(xxx, yyy, color=line_color, lw=2.5)
            else:
                ax.plot(xxx, yyy, color='white', lw=3)
                ax.plot(xxx, yyy, color='black', lw=2.5)

    # plot 45 degree line
    ax.plot(range_x, 1 - range_x, color='black', lw=1.0) #, zorder=0)
    ax.plot(range_x, 1 - range_x, color='lightgray', lw=0.75, alpha=1.0) #, zorder=0)

    # make colorbar pretty
    if (field != 'cross-time') and (field != 'cross'):
        bar_ticks = cbar.ax.get_xticks()
        dbar = bar_ticks[1] - bar_ticks[0]
        dbar_dec = get_precision(np.round(dbar, 7))
        bar_ticks_nice = ['%.*f' % (dbar_dec, bar_ticks[k]) for k in range(len(bar_ticks))]
        cbar.set_ticks(bar_ticks)
        cbar.set_ticklabels(bar_ticks_nice)
    cbar.ax.tick_params(length=3, width=1, labelsize=font_size)
    cbar.set_label(label_, size=font_size, labelpad=5)

    # edit axes ticks
    ax.tick_params(length=3, width=0.5, direction='in', labelsize=font_size)
    cbar.ax.tick_params(length=3, width=0.5, labelsize=font_size, pad=0)
    cbar.outline.set_linewidth(0.5)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(0.5)

    # fit line to threshold between unsuccessful and successful events
    # if fit_inv:
    #     # find threshold
    #     threshold = np.zeros((0, 2))
    #     for y in val_y:
    #         for x in val_x:
    #             qual = df.loc[(df[name_x] == x) & (df[name_y] == y)]['quality'].values
    #             if qual > 0.20:
    #                 threshold = np.append(threshold, [[x, y]], axis=0)
    #                 break
    #
    #     xdata = threshold[:, 0]
    #     ydata = threshold[:, 1]
    #
    #     # fit an x.y=cte line to test results
    #     function = inverse_fit
    #     init_est = 50
    #     fit_params, pcov = scipy.optimize.curve_fit(function, xdata, ydata, p0=init_est, maxfev=10000)
    #     residuals = ydata - function(xdata, *fit_params)
    #
    #     # find r2 of fit
    #     ss_res = np.sum(residuals ** 2)
    #     ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    #     r_squared = 1 - (ss_res / ss_tot)
    #     print('r2 = %f' % r_squared)
    #     print('params = %s' % fit_params)
    #     dx = val_x[1] - val_x[0]
    #     xrange = np.arange(val_x[0], val_x[-1] + dx, dx / 10)
    #
    #     # plot fitted line
    #     ax.plot(xrange, function(xrange, *fit_params), c='white', ls='--', lw=2)

    # apply stored limits
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)


    # if len(val_x) > 11:
    #     val_x = val_x[::2]
    ax.set_xticks(val_x)
    x_labels = []
    x_ticks = []
    for i in range(len(val_x)):
        if i % skip_xlabel == 0:
            if xlabel_dec > 0:
                val_x_print = int(val_x[i] * (10 ** xlabel_dec)) / (10 ** xlabel_dec)
            else:
                val_x_print = int(val_x[i])
            x_labels += [f'{val_x_print:,}' + xlabel_append]
            x_ticks += [val_x[i]]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels=x_labels, fontsize=font_size)

    # if len(val_y) > 11:
    #     val_y = val_y[::2]

    ax.set_yticks(val_y)
    y_labels = []
    y_ticks = []
    for i in range(len(val_y)):
        if i % skip_ylabel == 0:
            if i > 0:
                if ylabel_dec > 0:
                    val_y_print = int(val_y[i] * (10 ** ylabel_dec)) / (10 ** ylabel_dec)
                else:
                    val_y_print = int(val_y[i])
                y_labels += [f'{val_y_print:,}']
            else:
                y_labels += ['']
            y_ticks += [val_y[i]]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels=y_labels, fontsize=font_size)

    # apply stored limits
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # set aspect ratio
    # aspect = (x_lims[1] - x_lims[0])/(y_lims[1] - y_lims[0])
    # ax.set_aspect(aspect*len(val_y)/len(val_x))

    # set labels
    ax.set_xlabel(xlabel_, fontsize=font_size)
    ax.set_ylabel(ylabel_, fontsize=font_size)
    # ax.set_title(title_, fontsize=font_size, ha='center', va='center', pad=-30)

    # save results figure
    plt.savefig(group_path + '/0_group_square_%s.png' % field, dpi=600, bbox_inches='tight')
    plt.savefig(group_path + '/0_group_square_%s.svg' % field, dpi=600, bbox_inches='tight')


def get_square_group(group_options, n_asb=10, max_speed_inv=1.5, max_std=6,
                     ax_scale=1., plot_theory=False, max_cbar=None):

    group_path = group_options['output_dir'] + group_options['time_stamp']
    get_pivot(group_path, 'w_ff', 'w_rc', n_asb)

    for field in ['cross-time', 'std']:
        group_plot(group_path, field,
                   'w_ff', 'w_rc',
                   r'Feedforward ($F$)', r'Recurrent ($R$)',
                   n_asb=n_asb, max_speed_inv=max_speed_inv, max_std=max_std,
                   xlabel_dec=1, ylabel_dec=1, scale_x=ax_scale, scale_y=ax_scale, scale_z=1,
                   skip_xlabel=20, skip_ylabel=20,
                   plot_theory=plot_theory, max_cbar=max_cbar,
                   title_=r'')
                # , fit_inv=True)

    print('finished group plots for test %s' % group_options['time_stamp'])


def get_square_corr(group_options, n_asb,
                      plot_theory=False,
                      skip_xlabel=2, skip_ylabel=2,
                      xlabel_dec=0, ylabel_dec=1):
    group_path = group_options['output_dir'] + group_options['time_stamp']

    get_pivot(group_path, 'w_rc', 'w_ff', n_asb=n_asb)

    # read group results pivot
    file_path = group_path + '/0_group_square_pivot.txt'
    df = pd.read_csv(file_path)

    # plot results
    fig_height = 1.18
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    q=n_asb
    speeds = np.array(q/df['cross-time'].values)
    speeds = speeds[~np.isnan(speeds)]
    widths_all = np.array(df['std'].values)
    widths_all = widths_all[~np.isnan(widths_all)]

    # convert std to FWHM
    widths_all *= 2*np.sqrt(2*np.log(2))

    # limit analysis to widths smaller than 10 time steps
    widths = widths_all[widths_all <= 8]
    speeds = speeds[widths_all <= 8]

    corrfile = group_path + '/0_group_square_corr.txt'
    c, p = pearsonr(1/speeds, widths)
    xprint('Correlation: c = %.3f, p = %f' % (c, p), corrfile)

    # Fit the data to a line
    coefficients = np.polyfit(1/speeds, widths, 1)
    slope, intercept = coefficients

    # Generate the fitted line
    widths_fit = slope * (1/speeds) + intercept

    # Calculate R2
    ss_res = np.sum((widths - widths_fit) ** 2)
    ss_tot = np.sum((widths - np.mean(widths)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    xprint('R2: %f' % r_squared, corrfile)
    xprint('slope: %f' % slope, corrfile)
    xprint('intercept: %s' % intercept, corrfile)
    # print('max time = ', 1/slope)
    # print('tau0 = ', intercept/slope)

    x_array = np.arange(0, np.max(1/speeds), 0.1)
    ax.scatter(1/speeds, widths, s=1, c='k')
    ax.plot(x_array, slope * x_array + intercept, ls='--', lw=2, c='gray')

    # edit axes
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

    plt.savefig(group_path + '/0_group_square_corr.png', dpi=600, bbox_inches='tight')
    plt.savefig(group_path + '/0_group_square_corr.svg', dpi=600, bbox_inches='tight')
