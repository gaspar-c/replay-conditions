import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, rc, patches
from matplotlib.colors import LogNorm
import pandas as pd
from my_code.discrete_model import simplified_theory
from scipy.stats import pearsonr
from my_code.aux_functions import xprint

rc('text', usetex=True)
rc('mathtext', fontset='stix')
rc('font', family='sans-serif')


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


def get_replay_pivot(group_path, var1, var2, qual_thres=0.0, act_low_thres=0.0, act_up_thres=2.0,
                     min_speed=0, savefile=True, prints=True):
    """
    """
    # read test results file and create a DataFrame
    file_path = group_path + '/0_group_replay.txt'
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
    if 'explosion' in df:
        df['explosion'] = bool_from_str(df['explosion'])
    for col_name in ['avg_speed', 'asy_speed', 'asy_act', 'asy_width']:
        df[col_name] = float_from_str(df[col_name])

    df['asy_speed'] = 1 / df['asy_speed']
    df['avg_speed'] = 1 / df['avg_speed']

    # cut success if activity is below threshold:
    df.loc[df['asy_act'] <= act_low_thres, 'replay'] = False
    df.loc[df['asy_act'] >= act_up_thres, 'replay'] = False

    # replace failed replays with NaNs, so they don't count towards mean results
    for col_name in ['asy_speed', 'asy_act', 'asy_width', 'avg_speed']:
        if col_name in df:
            df.loc[df['replay'] == False, col_name] = np.nan

    for field in ['asy_speed', 'asy_width', 'avg_speed']:
        df.loc[df['asy_speed'] < min_speed, field] = np.nan

    if prints:
        print(df)

    # create pivot table that counts successful replays for each (var1, var2) pair
    if 'explosion' in df:
        replay_pivot = df.pivot_table(values=['replay', 'explosion', 'avg_speed', 'asy_speed', 'asy_act', 'asy_width'],
                                      columns=[var1, var2],
                                      aggfunc={'replay': [np.sum, 'size'],
                                               'explosion': np.sum,
                                               'avg_speed': np.mean,
                                               'asy_speed': np.mean,
                                               'asy_act': np.mean,
                                               'asy_width': np.mean,}).T
    else:
        replay_pivot = df.pivot_table(values=['replay', 'avg_speed', 'asy_speed', 'asy_act', 'asy_width'],
                                      columns=[var1, var2],
                                      aggfunc={'replay': [np.sum, 'size'],
                                               'avg_speed': np.mean,
                                               'asy_speed': np.mean,
                                               'asy_act': np.mean,
                                               'asy_width': np.mean,}).T

    # check that all tests are complete
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

    # calculate replay quality (percentage of success)
    replay_pivot['quality'] = replay_pivot['replay']['sum'] / max_trials
    # replay_pivot['replay'].drop(labels='sum', axis=2, inplace=True)
    replay_pivot.drop(labels='replay', axis=1, inplace=True)
    replay_pivot.reset_index(inplace=True)

    # cut counts if quality is below threshold:
    for field_ in ['avg_speed', 'asy_speed', 'asy_width', 'asy_act']:
         replay_pivot.loc[replay_pivot['quality'] < qual_thres, field_] = np.nan

    # flatten header
    flat_cols = []
    for i in replay_pivot.columns:
        flat_cols.append(i[0])
    replay_pivot.columns = flat_cols

    if prints:
        print(replay_pivot.to_string(index=False))

    if savefile:
        replay_pivot.to_csv(group_path + '/0_group_replay_pivot.txt', index=False)

    return replay_pivot


def group_plot(group_path, field,
               name_x, name_y, xlabel_, ylabel_,
               label_='',
               scale_x=1, scale_y=1, skip_xlabel=2, skip_ylabel=2,
               xlabel_append='', fit_inv=False, theor_plot=False, v_file=None,
               log_scale=False, time_step=None, dist_width=None, full_square=True,
               min_speed=None, cond2=None, cond3=None,
               cbar_ticks_=[], cbar_ticklabels_=[], vspan_=[],
               square_coords=[],
               color_map='gist_heat'):
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

    """

    # read group results pivot
    replay_file_path = group_path + '/0_group_replay_pivot.txt'
    df_replay = pd.read_csv(replay_file_path)

    # scale values
    df_replay[name_x] = df_replay[name_x] * scale_x
    df_replay[name_y] = df_replay[name_y] * scale_y

    # sort values
    val_x = np.sort(np.unique(df_replay[name_x].values))
    val_y = np.sort(np.unique(df_replay[name_y].values))

    # get percentage of successful replays for each (x,y) value pair
    xx, yy = np.meshgrid(val_x, val_y, indexing='xy')
    zz = np.zeros_like(xx)
    zz_explosion = np.zeros_like(xx)
    for i in range(len(val_x)):
        for j in range(len(val_y)):
            if 'explosion' in df_replay:
                z_explosion = df_replay.loc[(df_replay[name_x] == val_x[i]) & (df_replay[name_y] == val_y[j])]['explosion'].values
                if z_explosion[0] > 0:
                    zz_explosion[j, i] = 1.0

            z = df_replay.loc[(df_replay[name_x] == val_x[i]) & (df_replay[name_y] == val_y[j])][field].values
            if len(z) == 1:
                zz[j, i] = z[0]

    dx = val_x[1] - val_x[0]
    x_coord = np.append(val_x, val_x[-1] + dx) - dx / 2

    dy = val_y[1] - val_y[0]
    y_coord = np.append(val_y, val_y[-1] + dy) - dy / 2

    # plot results
    fig_height = 1.18
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if len(vspan_) > 0:
        vmin_ = vspan_[0]
        vmax_ = vspan_[1]
    else:
        vmin_ = np.nanmin(zz)
        vmax_ = np.nanmax(zz)

    if log_scale:
        norm = LogNorm(vmin=vmin_,
                       vmax=vmax_
                       )
        color_plot = ax.pcolormesh(x_coord, y_coord, zz, norm=norm, cmap=color_map, rasterized=True)
    else:
        color_plot = ax.pcolormesh(x_coord, y_coord, zz,
                                   vmin=vmin_, vmax=vmax_,
                                   cmap=color_map, rasterized=True)

    cbar = fig.colorbar(color_plot, ax=ax, location='top', pad=0.02)

    # plot explosions
    zz_explosion_masked = np.ma.masked_where (zz_explosion == 0, zz_explosion)
    ax.pcolormesh(x_coord, y_coord, zz_explosion_masked, cmap='gray', alpha=0.5, rasterized=False)

    # grey out outside min-max range
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

    # add lines and squares for specific coordinates
    for (x_, y_, color_) in square_coords:
        ax.add_patch(patches.Rectangle((x_ - dx / 2, y_ - dy / 2), dx, dy,
                                edgecolor=color_, fill=None, lw=0.5))
        ax.plot([x_ - 0.9 * dx / 2, x_ + 0.9 * dx / 2], [y_ - 0.9 * dy / 2, y_ + 0.9 * dy / 2], color=color_, lw=0.5)
        ax.plot([x_ - 0.9 * dx / 2, x_ + 0.9 * dx / 2], [y_ + 0.9 * dy / 2, y_ - 0.9 * dy / 2], color=color_, lw=0.5)
        line_ = df_replay.loc[np.isclose(df_replay['p_ff'], x_) & np.isclose(df_replay['p_rc'], y_)][field].values[0]
        cbar.ax.axvline(line_, c='k', lw=2)
        cbar.ax.axvline(line_, c=color_, lw=1)

    if len(cbar_ticks_) > 0:
        cbar.set_ticks(cbar_ticks_)
        if len(cbar_ticklabels_) > 0:
            cbar.set_ticklabels(cbar_ticklabels_)
        else:
            cbar.set_ticklabels(cbar_ticks_)

    ax.set_facecolor('darkgray')

    if full_square:
        ax.set_xlim(val_x[0]-dx/2, val_x[-1]+dx/2)
        ax.set_ylim(val_y[0]-dy/2, val_y[-1]+dy/2)
    else:
        ax.set_xlim(val_x[0], val_x[-1])
        ax.set_ylim(val_y[0], val_y[-1])

    # store axis limits
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    # cbar.set_label(label_, size=font_size, rotation=-90, va='top', labelpad=30)

    if (field == 'asy_speed') and (time_step is not None) and theor_plot:
        n_asb = 10
        square_width = dist_width
        print(square_width)
        with open(group_path + '/0_group_log.log', 'a') as file:
            file.write('min %s: %f\n' % (field, np.nanmin(zz)))
            file.write('max %s: %f\n' % (field, np.nanmax(zz)))

        # s0 = np.min([min_speed * time_step, 1])
        if min_speed is None:
            s0 = np.nanmin(zz) * time_step
        else:
            s0 = min_speed * time_step

        # Define the step size for the x and y values
        step_size_x = 0.01
        step_size_y = 0.05
        # Create a new range of x and y values with the defined step sizeww
        range_x = np.arange(step_size_x, val_x[-1] + dx, step_size_x) / square_width
        range_y = np.arange(step_size_y, val_y[-1] + 2 * dy, step_size_y) / square_width

        # plot q=inf
        theor_check = simplified_theory(range_x, range_y, n_asb, None, q_inf=True, speed=s0)
        theor_true = np.argwhere(theor_check==True)
        unique, indices = np.unique(theor_true[:, 1], return_index=True)
        xxx = range_x[theor_true[indices][:, 0]] * square_width
        yyy = range_y[theor_true[indices][:, 1]] * square_width
        ax.plot(xxx, yyy, color='white', lw=1.0, ls='--')

    # plot cond2
    if cond2 is not None:
        range_x = np.arange(0, cond2 + dx , dx)
        ax.plot(range_x, (cond2 - range_x), color='white', lw=1.0, ls='--')

    # plot cond3
    if cond3 is not None:
        ax.axvline(cond3, color='white', lw=1.0, ls='--')

    # apply stored limits
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # edit axes ticks
    ax.tick_params(length=1.5, width=0.5, direction='out', labelsize=font_size)
    cbar.ax.tick_params(length=1.5, width=0.5, labelsize=font_size, pad=0)
    cbar.outline.set_linewidth(0.5)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(0.5)

    ax.set_xticks(val_x)
    x_labels = []
    x_ticks = []
    for i in np.arange(0, len(val_x), skip_xlabel):
        x_ticks += [val_x[i]]
        x_labels += [f'{int(val_x[i]):,}' + xlabel_append]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels=x_labels, fontsize=font_size)

    ax.set_yticks(val_y)
    y_labels = ['0']
    y_ticks = [0]
    for i in np.arange(skip_ylabel, len(val_y), skip_ylabel):
        y_ticks += [val_y[i]]
        y_labels += [f'{int(val_y[i]):,}']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels=y_labels, fontsize=font_size)

    # apply stored limits
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)

    # set labels
    ax.set_xlabel(xlabel_, fontsize=font_size)
    ax.set_ylabel(ylabel_, fontsize=font_size)
    ax.set_title(label_, fontsize=font_size, pad=20)

    # save results figure
    figname = group_path + '/0_group_results12_%s' % field
    if min_speed is not None:
        figname += '_lim_%.2f' % min_speed
    plt.savefig(figname + '.png', dpi=600, bbox_inches='tight')
    plt.savefig(figname + '.svg', dpi=600, bbox_inches='tight')

    plt.close()



def get_correlations(group_options, ):
    group_path = group_options['output_dir'] + group_options['time_stamp']
    get_replay_pivot(group_path, 'p_ff', 'p_rc', qual_thres=0.8,
                     # act_low_thres=0.98, act_up_thres=1.02)
                    act_low_thres=0.90, act_up_thres=1.10)

    # read group results pivot
    replay_file_path = group_path + '/0_group_replay_pivot.txt'
    df_replay = pd.read_csv(replay_file_path)


    # plot results
    fig_height = 1.18
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    speeds = np.array(df_replay['asy_speed'].values)
    speeds = speeds[~np.isnan(speeds)]
    widths = np.array(df_replay['asy_width'].values)
    widths = widths[~np.isnan(widths)]

    corrfile = group_path + '/0_group_results12_corr.txt'
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

    x_array = np.arange(0, 10.5, 0.1)
    ax.scatter(1/speeds, widths, s=1, c='k')
    ax.plot(x_array, slope * x_array + intercept, ls='--', lw=2, c='gray')

    # edit axes
    ax.tick_params(length=3, width=0.5, direction='out', labelsize=font_size)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.5)
    for spines in ['top', 'right']:
        ax.spines[spines].set_visible(False)

    ax.set_xlabel(r'Speed$^{-1}$ (ms)', fontsize=font_size)
    ax.set_ylabel('FWHM (ms)', fontsize=font_size)
    ax.set_ylim([4, 10.5])
    ax.set_xlim([0, 10.5])
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '5', '10'], fontsize=font_size)
    ax.set_yticks([4, 7, 10])
    ax.set_yticklabels(['4', '7', '10'], fontsize=font_size)

    # save results figure
    plt.savefig(group_path + '/0_group_results12_corr.png', dpi=600, bbox_inches='tight')
    plt.savefig(group_path + '/0_group_results12_corr.svg', dpi=600, bbox_inches='tight')


def get_speed_lines(group_options, time_step, dist_width, speed_lims=[None]):
    group_path = group_options['output_dir'] + group_options['time_stamp']

    # read group results pivot
    replay_file_path = group_path + '/0_group_replay_pivot.txt'
    pivot_none = pd.read_csv(replay_file_path)

    # sort values
    val_x = np.sort(np.unique(pivot_none['p_ff'].values))
    val_y = np.sort(np.unique(pivot_none['p_rc'].values))
    dx = val_x[1] - val_x[0]
    dy = val_y[1] - val_y[0]

    # plot results
    fig_height = 1.47
    fig_width = fig_height / 1.2
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.set_facecolor('darkgray')

    color_norm = mpl.colors.Normalize(vmin=0, vmax=0.35)
    cmap = mpl.cm.get_cmap('gist_heat')

    cmap_bg = mpl.colors.ListedColormap(['darkgray'])
    color_norm_bg = mpl.colors.Normalize(vmin=0, vmax=0.4)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=color_norm_bg, cmap=cmap_bg), ax=ax, location='top',
                        pad=0.02, shrink=0.99)

    ax.set_xlim(val_x[0], val_x[-1])
    ax.set_ylim(val_y[0], val_y[-1])

    # store axis limits
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    cbar_ticks_ = [0.1, 0.2, 0.3]
    cbar_ticklabels_ = ['0.1', '0.2', '0.3']
    # Draw the lower and left edges of the border squares for specific values where zz_plot values decrease
    for k in range(len(speed_lims)):
        min_speed_ = speed_lims[k]

        if min_speed_ is None:
            df_replay = pivot_none

        else:
            df_replay = get_replay_pivot(group_path, 'p_ff', 'p_rc', qual_thres=0.8,
                                         act_low_thres=0.90, act_up_thres=1.10,
                                         min_speed=min_speed_, savefile=False, prints=False)

        xx, yy = np.meshgrid(val_x, val_y, indexing='xy')
        zz = np.zeros_like(xx)
        for i in range(len(val_x)):
            for j in range(len(val_y)):
                z = df_replay.loc[(df_replay['p_ff'] == val_x[i]) & (df_replay['p_rc'] == val_y[j])]['asy_speed'].values
                if len(z) == 1:
                    zz[j, i] = z[0]

        if min_speed_ is None:
            min_speed_ = np.nanmin(zz)
            color_ = 'black'
        else:
            color_ = cmap(color_norm(min_speed_))

        if k < len(speed_lims) - 1:
            cbar.ax.add_patch(patches.Rectangle(
                (min_speed_, 0),
                speed_lims[k+1] - min_speed_, 1,
                lw=0, color=color_, alpha=1.0))
        else:
            max_speed_ = np.nanmax(zz)
            cbar.ax.add_patch(patches.Rectangle(
                (min_speed_, 0),
                max_speed_ - min_speed_, 1,
                lw=0, color=color_, alpha=1.0))

        x_coord = np.append(val_x, val_x[-1] + dx) - dx / 2
        y_coord = np.append(val_y, val_y[-1] + dy) - dy / 2
        ax.pcolormesh(x_coord, y_coord, np.ma.masked_where(np.isnan(zz) | (zz <= 0), zz),
                      cmap=mpl.colors.ListedColormap([color_]), rasterized=True)

        # plot analytical line
        s0 = np.min([min_speed_ * time_step, 1])
        # Create a new range of x and y values with the defined step size
        step_size_x = 0.01 / 100
        step_size_y = 0.05 / 100
        range_x = np.arange(step_size_x, val_x[-1] + dx, step_size_x) * 100 / dist_width
        range_y = np.arange(step_size_y, val_y[-1] + 2 * dy, step_size_y) * 100 / dist_width
        theor_check = simplified_theory(range_x, range_y, np.nan, np.nan, q_inf=True, speed=s0)
        theor_true = np.argwhere(theor_check == True)
        unique, indices = np.unique(theor_true[:, 1], return_index=True)
        xxx = range_x[theor_true[indices][:, 0]] * dist_width / 100
        yyy = range_y[theor_true[indices][:, 1]] * dist_width / 100
        ax.plot(xxx, yyy, color='black', lw=1.5, ls='-')
        ax.plot(xxx, yyy, color=color_, lw=1, ls='-')

    cbar.set_ticks(cbar_ticks_)
    cbar.set_ticklabels(cbar_ticklabels_)

    # edit axes ticks
    ax.tick_params(length=2.0, width=0.5, direction='out', labelsize=font_size, pad=2)
    cbar.ax.tick_params(length=2.0, width=0.5, labelsize=font_size, pad=0)

    cbar.outline.set_linewidth(0.5)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_linewidth(0.5)

    skip_xlabel = 4
    skip_ylabel = 4

    ax.set_xticks(val_x)
    x_labels = []
    x_ticks = []
    for i in np.arange(0, len(val_x), skip_xlabel):
        x_ticks += [val_x[i]]
        x_labels += [f'{int(val_x[i]*100):,}']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels=x_labels, fontsize=font_size)

    ax.set_yticks(val_y)
    y_labels = ['0']
    y_ticks = [0]
    for i in np.arange(skip_ylabel, len(val_y), skip_ylabel):
        y_ticks += [val_y[i]]
        y_labels += [f'{int(val_y[i]*100):,}']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels=y_labels, fontsize=font_size)

    # set aspect ratio
    aspect = (x_lims[1] - x_lims[0])/(y_lims[1] - y_lims[0])
    ax.set_aspect(aspect*len(val_y)/len(val_x))

    # set labels
    ax.set_xlabel(r'$p_f$ (\%)', fontsize=font_size)
    ax.set_ylabel(r'$p_r$ (\%)', fontsize=font_size)
    ax.set_title(r'Speed (1 / ms)', fontsize=font_size, pad=20)
    # , loc='right', va='bottom', pad=1)

    # save results figure
    figname = group_path + '/0_group_results12_speed_lim'
    plt.savefig(figname + '.png', dpi=600, bbox_inches='tight')
    plt.savefig(figname + '.svg', dpi=600, bbox_inches='tight')

    plt.close()

    print('finished group plots for test %s' % group_options['time_stamp'])






