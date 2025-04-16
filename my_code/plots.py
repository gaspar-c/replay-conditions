from brian2 import *
from matplotlib import colors
import colorsys
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.pyplot as plt
import math
import numpy as np
from my_code.network import VoltageStimulus
from my_code.aux_functions import (trim_brian_monitor, filter_array, check_brian_monitor,
                                   calc_low_pass_filter, calc_rate_monitor, calc_network_frequency,
                                   trim_brian_monitor_group)


def color_gradient(color_i, color_f, n):
    """
    calculates array of intermediate colors between two given colors

    Args:
        color_i: initial color
        color_f: final color
        n: number of colors

    Returns:
        color_array: array of n colors
    """

    if n > 1:
        rgb_color_i = np.array(colors.to_rgb(color_i))
        rgb_color_f = np.array(colors.to_rgb(color_f))
        color_array = [None] * n
        for i in range(n):
            color_array[i] = colors.to_hex(rgb_color_i*(1-i/(n-1)) + rgb_color_f*i/(n-1))
    else:
        color_array = [color_f]

    return color_array


def darken_color(color, factor=0.):
    """
    darkens the given input color by a given factor
    Args:
        color: input color
        factor: darkening factor

    Returns:
        darkened color
    """

    try:
        color_name = colors.cnames[color]
    except:
        color_name = color

    c_params = colorsys.rgb_to_hls(*colors.to_rgb(color_name))

    return colorsys.hls_to_rgb(c_params[0], factor * c_params[1], c_params[2])


def get_height(plot_height):
    """
    converts a given plot height to a round
    number with corresponding axis ticks
    """
    if plot_height > 10:
        y_ticks = [0, int(math.ceil(plot_height / 20) * 10)]
    elif plot_height > 5:
        y_ticks = [0, int(math.ceil(plot_height / 10) * 5)]
    elif plot_height > 1:
        y_ticks = [0, int(math.ceil(plot_height / 4) * 2)]
    elif plot_height > 0.5:
        y_ticks = [0.0, 0.5]
    elif plot_height > 0.1:
        y_ticks = [0.0, 0.1]
    elif plot_height > 0.05:
        y_ticks = [0, 0.05]
    elif plot_height > 0.01:
        y_ticks = [0, 0.01]
    elif plot_height > 0.005:
        y_ticks = [0, 0.005]
    else:
        y_ticks = [0, 0.001]

    if plot_height == 0:
        y_ticks = [0]
    # else:
    #     y_ticks = [0, 1]
    #     plot_height = 1

    return plot_height, y_ticks


class MySubplot:
    """
    subplot object
    """
    def __init__(self, subplot_name, monitors, plot_type, plot_colors, plot_params,
                 x_label, height_ratio, n_raster=250):
        """
        Args:
            subplot_name: subplot name
            plot_type: type of plot
                        'raster': raster plot
                        'trace': "normal" 2d plot
                        'density': 2D histogram
                        'hist': histogram
                        ...
            plot_colors: array of colors for each monitor to be plotted on this subplot
            plot_params: plot parameters
            x_label: x-axis label
            height_ratio: subplot height ratio
            n_raster: number of neurons in raster plot
        """

        self.subplot_name = subplot_name
        self.monitors = monitors
        self.num_traces = len(self.monitors)
        self.plot_type = plot_type
        self.plot_colors = plot_colors
        self.x_label = x_label
        self.ax = None
        self.lines = [None] * self.num_traces
        self.font_size = plot_params['text_font'].get_param()
        self.spine_width = plot_params['spine_width'].get_param()
        self.height_ratio = height_ratio
        self.n_raster = n_raster

        self.fixed_y_ticks = None
        self.fixed_y_lims = None
        self.fixed_y_ticklabels = None

        self.fixed_x_ticks = None
        self.fixed_x_lims = None
        self.y_log = False

    def attr_ax(self, subplot_axes):
        self.ax = subplot_axes

    def set_title(self, title_text):
        self.ax.set_title(title_text, loc='left', x=0., y=1.02, fontsize=self.font_size)

    def hide_bottom(self):
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])

    def hide_left(self):
        self.ax.spines['left'].set_visible(False)
        self.ax.set_yticks([])

    def show_time_labels(self):
        self.ax.set_xlabel('time (s)')

    def hide_time_labels(self):
        self.ax.set_xticklabels(labels=[])
        self.ax.set_xlabel('')

    def fix_y_axis(self, y_lims, y_ticks, ticklabels=None):
        self.fixed_y_ticks = y_ticks
        self.fixed_y_lims = y_lims
        if ticklabels is not None:
            self.fixed_y_ticklabels = ticklabels

    def fix_x_axis(self, x_lims, x_ticks):
        self.fixed_x_ticks = x_ticks
        self.fixed_x_lims = x_lims

    def set_x_ticks(self, x_ticks):
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels(labels=x_ticks, fontsize=self.font_size)

    def set_y_ticks(self, y_ticks):
        self.ax.set_yticks(y_ticks)

        if not self.y_log:
            self.ax.set_yticklabels(labels=y_ticks, fontsize=self.font_size)
        else:
            log_ylabels = [None] * len(y_ticks)
            for i in range(len(y_ticks)):
                log_ylabels[i] = '10$^{%s}$' % '{:+}'.format(int(np.log10(y_ticks[i])))
            self.ax.set_yticklabels(labels=log_ylabels, fontsize=self.font_size)

    def set_y_label(self, label):
        self.ax.set_ylabel(label, fontsize=self.font_size)

    # def pretty_ticks(self):
    #     y_ticks = self.ax.get_yticks()
    #     self.set_y_ticks(y_ticks)
    #     x_ticks = self.ax.get_xticks()
    #     self.set_x_ticks(x_ticks)

    def set_time_ticks(self, plot_start, plot_stop, n_ticks):
        time_ticks = [round(plot_start + (plot_stop - plot_start) * i / n_ticks, 3) for i in
                      range(n_ticks + 1)]
        self.set_x_ticks(time_ticks)

    def add_lines(self):
        for i in range(self.num_traces):
            if self.plot_type == 'raster':
                self.lines[i], = self.ax.plot([], [], '.', ms=3.0, color=self.plot_colors[i], markeredgewidth=0.0)

            elif self.plot_type == 'trace':
                self.lines[i], = self.ax.plot([], [], lw=1, color=self.plot_colors[i])

    def general_format(self):

        self.ax.tick_params(top=False, which='both', labelsize=self.font_size, direction='out', width=self.spine_width)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # all raster plots include only n_raster*height_ratio neurons:
        if self.plot_type == 'raster':
            self.ax.set_ylim([-2, int(self.n_raster) + 1])
            self.hide_left()

            # if more than one monitor in one raster plot (for assemblies), split the n_raster*height_ratio neurons equally:
            if self.num_traces > 1:
                self.ax.set_ylim([-2, int(self.n_raster * self.height_ratio) + 1])
                asb_height = int(self.n_raster * self.height_ratio / self.num_traces)
                h_lines = np.arange(0, asb_height*(self.num_traces + 1), asb_height)
                for h in h_lines:
                    self.ax.axhline(h - 0.5, lw=0.5, color='lightgray', ls='solid')

        if self.fixed_y_ticks is not None:
            self.set_y_ticks(self.fixed_y_ticks)
        if self.fixed_y_lims is not None:
            self.ax.set_ylim(self.fixed_y_lims)
        if self.fixed_y_ticklabels is not None:
            self.ax.set_yticklabels(self.fixed_y_ticklabels)

        if self.fixed_x_ticks is not None:
            self.set_x_ticks(self.fixed_x_ticks)
        if self.fixed_x_lims is not None:
            self.ax.set_xlim(self.fixed_x_lims)

        if self.y_log:
            self.ax.set_yscale('log')
            self.ax.set_ylim([10**-2, None])

        # if fixed x ticks and limits have been set, use those:
        if self.fixed_x_ticks is not None:
            self.set_x_ticks(self.fixed_x_ticks)
        if self.fixed_x_lims is not None:
            self.ax.set_xlim(self.fixed_x_lims)

        # if any x label has been set, use it:
        if self.x_label != '':
            self.ax.set_xlabel(self.x_label, fontsize=self.font_size)

        # self.pretty_ticks()

    def set_lines(self):
        """
        attribute the correct monitor to each subplot line
        """

        # iterate through each line to be plotted
        max_height = 0
        for i in range(self.num_traces):

            # multiple raster plots in one subplot are split equally:
            if (self.plot_type == 'raster'):  # and (self.num_traces > 1):
                if (self.num_traces > 1):
                    num = int(self.n_raster * self.height_ratio / self.num_traces)
                else:
                    num = self.n_raster
                times, neurons = self.monitors[i]
                idx = neurons < num
                shifted_neurons = neurons[idx] + i*num
                monitor = times[idx], shifted_neurons
            else:
                monitor = self.monitors[i]

            # set monitor x and y arrays to subplot line
            if self.plot_type in ['raster', 'trace']:
                self.lines[i].set_xdata(monitor[0])
                self.lines[i].set_ydata(monitor[1])

            # auto update y axis for trace plots:
            if self.plot_type == 'trace':
                max_val = np.max(monitor[1])
                if max_val > max_height:
                    max_height = max_val
                    ax_height, y_ticks = get_height(max_val)
                    if self.fixed_y_lims is None:
                        if ax_height > 0:
                            self.ax.set_ylim([0, ax_height * 1.05])
                        else:
                            self.ax.set_ylim([-1, 1])
                    if self.fixed_y_ticks is None:
                        self.set_y_ticks(y_ticks)

            # create histogram plots:
            if self.plot_type == 'hist':
                self.ax.hist(monitor, bins=30, color=self.plot_colors, density=False)

                if self.fixed_y_ticks is None:
                    n_y_ticks = 2
                    _, top = self.ax.get_ylim()
                    y_ticks = np.unique([round((top * i / n_y_ticks), 2) for i in range(n_y_ticks + 1)])
                    self.set_y_ticks(y_ticks)

            if self.plot_type == 'density':
                x_len = len(monitor[0])
                y_len = len(monitor[1])

                # 2d histogram
                x_array = np.tile(monitor[0], (y_len, 1)).flatten()
                y_array = monitor[1].flatten()
                heights, x_edges, y_edges = np.histogram2d(x_array, y_array, bins=[x_len, 50])
                self.ax.pcolormesh(x_edges, y_edges, heights.T, cmap='Greys', rasterized=True, vmax=int(y_len)*0.10)

                # # mean and std
                # mean_array = np.mean(monitor[1], axis=0)
                # std_array = np.std(monitor[1], axis=0)
                #
                # self.ax.plot(monitor[0], mean_array, ls='-', color='black', lw=1, alpha=0.5)
                # self.ax.plot(monitor[0], mean_array + std_array, ls='--', color='black', lw=1, alpha=0.5)
                # self.ax.plot(monitor[0], mean_array - std_array, ls='--', color='black', lw=1, alpha=0.5)

    def add_vlines(self, x_vals):
        for x_val in x_vals:
            self.ax.axvline(x_val, c='gray', ls='--', lw=1)


class MySubplotGroup:
    """
    object with group of subplots
    """
    def __init__(self, group_name, group_title, group_type, hide_time_ax=False, y_label=['']):
        """
        Args:
            group_name: subplot group name
            group_title: subplot group title
            group_type: subplot group type
                        'time': subplot group where time is on the x-axis
                        ...: other
            hide_time_ax: [True/False], may hide time axis
            y_label: subplot group y-axis label
        """

        self.group_name = group_name
        self.group_title = group_title
        self.group_type = group_type
        self.hide_time_ax = hide_time_ax
        self.time_labels = False
        self.y_label = y_label
        self.subplots = []

    def add_subplot(self, subplot_name, monitors, plot_type, plot_colors, plot_params,
                    x_label='', height_ratio=1.0, n_raster=250):
        """
        add a subplot to this subplot group.
        each subplot within the group can have more than one monitor.

        Args:
            subplot_name: name of subplot
            monitors: array with brian monitors to be plotted
            plot_type: type of subplot
                        'raster': raster plot
                        'trace': "normal" 2d plot
                        'density': 2D histogram
                        'hist': histogram
                        ...
            plot_colors: array with colors for each monitor to be plotted
            plot_params: plot parameters
            x_label: label for subplot x-axis
            height_ratio: subplot height ratio
            n_raster: number of neurons in raster plot
        """

        self.subplots.append(MySubplot(subplot_name, monitors,
                                       plot_type, plot_colors, plot_params,
                                       x_label, height_ratio, n_raster))

    def get_num_vert(self):
        """
        get total number of rows of subplots in this group
        """
        if self.group_type == 'time':
            return len(self.subplots)
        else:
            return 1

    def get_num_horiz(self):
        """
        get total number of columns of subplots in this group
        ('time' group can only have one column)
        """

        if self.group_type != 'time':
            return len(self.subplots)
        else:
            return 1

    def get_height_ratios(self, i):
        """
        get array with subplot height ratios
        """
        return self.subplots[i].height_ratio

    def reveal_time_labels(self):
        """
        show/hide time labels if it is/isn't a 'time' group
        """

        if self.group_type == 'time':
            if self.time_labels and not self.hide_time_ax:
                self.subplots[-1].show_time_labels()
            else:
                self.subplots[-1].hide_time_labels()

    def init_group_format(self, time_labels=False):
        self.time_labels = time_labels
        self.subplots[0].set_title(self.group_title)

        # for every subplot in group:
        for i in range(len(self.subplots)):
            self.subplots[i].general_format()

            # hide bottom of all subplots in 'time' group except for the last:
            if self.group_type == 'time':
                if i < len(self.subplots) - 1:
                    self.subplots[i].hide_bottom()
                if self.hide_time_ax:
                    self.subplots[-1].hide_bottom()

        # if a y label has given, set it on the middle subplot of the group:
        if len(self.y_label) == 2:
            self.subplots[0].ax.set_ylabel(self.y_label[0])
            self.subplots[1].ax.set_ylabel(self.y_label[1])
        else:
            if self.y_label != ['']:
                if self.group_type == 'time':
                    label_idx = int(len(self.subplots) / 2)
                else:
                    label_idx = 0
                self.subplots[label_idx].set_y_label(self.y_label[0])

        self.reveal_time_labels()

    def set_time_axes(self, plot_start, plot_stop, n_ticks):
        if self.group_type == 'time':
            for i in range(len(self.subplots)):
                self.subplots[i].ax.set_xlim([plot_start, plot_stop])

            if not self.hide_time_ax:
                self.subplots[-1].set_time_ticks(plot_start, plot_stop, n_ticks)
                self.reveal_time_labels()


class PlotRaster:
    """
    object to create raster plots of simulation
    """
    def __init__(self, pops=[], plot_bg=True,bg_in=True):
        """
        Args:
            pops: list of neuron populations to plot
            plot_bg: plot background cells
            bg_in: if there are assemblies, include bg neurons in same raster
        """
        self.pops = pops
        self.plot_bg = plot_bg
        self.bg_in = bg_in
        self.monitor_types = ['spike']

    def perform_plot(self, network, settings, test_range):
        use_tex = settings.options['use_tex']
        hide_time_ax = settings.plot_params['hide_time_ax'].get_param()
        n_asb = settings.net_objects.n_asb

        # create group of raster plots:
        if use_tex:
            group_name = r'\textbf{Spikes}'
        else:
            group_name = 'Spikes'
        subplot_group = MySubplotGroup('spm', group_name, 'time', hide_time_ax=hide_time_ax)

        for pop in self.pops:
            monitors = []
            if isinstance(network['pop_' + pop.name], NeuronGroup) and pop.asb_flag:
                for i in range(n_asb):
                    spm = 'spm_' + pop.name + '_asb_' + str(i + 1)
                    monitors += [trim_brian_monitor(network[spm], network[spm].i, 1, test_range[0], test_range[1])]

                colors_spm = color_gradient(pop.plot_color, darken_color(pop.plot_color, 0.3), n_asb)

                spm_out = 'spm_' + pop.name + '_out'
                if spm_out in network and self.plot_bg:
                    if self.bg_in:
                        colors_spm += ['black']
                        monitors += [trim_brian_monitor(network[spm_out], network[spm_out].i, 1,
                                                       test_range[0], test_range[1])]

                        subplot_group.add_subplot('spm_' + pop.name, monitors, 'raster', colors_spm,
                                                  settings.plot_params, height_ratio=pop.raster_height,
                                                  n_raster=pop.n_raster)

                    else:
                        monitor_out = [trim_brian_monitor(network[spm_out], network[spm_out].i, 1,
                                                      test_range[0], test_range[1])]

                        subplot_group.add_subplot('spm_' + pop.name + '_out', monitor_out, 'raster',
                                                  ['black'], settings.plot_params, height_ratio=1.0,
                                                  n_raster=int(pop.n_raster / pop.raster_height))

                        subplot_group.add_subplot('spm_' + pop.name, monitors, 'raster', colors_spm,
                                                  settings.plot_params, height_ratio=pop.raster_height,
                                                  n_raster=pop.n_raster)



                else:
                    subplot_group.add_subplot('spm_' + pop.name, monitors, 'raster', colors_spm,
                                              settings.plot_params, height_ratio=pop.raster_height,
                                              n_raster=pop.n_raster)

            else:
                spm = 'spm_' + pop.name
                monitors += [trim_brian_monitor(network[spm], network[spm].i, 1, test_range[0], test_range[1])]
                subplot_group.add_subplot(spm, monitors, 'raster', [pop.plot_color],
                                          settings.plot_params, height_ratio=pop.raster_height, n_raster=pop.n_raster)

        return subplot_group


class PlotPopRate:
    """
    object to create population rate plots of simulation
    """
    def __init__(self, pops=[], bg_in=True, overall=False, filter_func='gauss', filter_width=0*ms, y_max=None, title=None):
        """
        Args:
            pops: list of neuron populations to plot
            bg_in: if there are assemblies, include bg neurons in same subplot
            filter_func: kernel function to filter population rate
            filter_width: kernel width
            y_max: maximum value on y-axis
        """
        self.pops = pops
        self.bg_in = bg_in
        self.overall = overall
        self.filter_func = filter_func
        self.filter_width = filter_width
        self.y_max = y_max
        self.title = title
        self.monitor_types = ['spike']

    def perform_plot(self, network, settings, test_range):
        use_tex = settings.options['use_tex']
        hide_time_ax = settings.plot_params['hide_time_ax'].get_param()
        n_asb = settings.net_objects.n_asb

        sim_dt = settings.sim_params['sim_dt'].get_param()

        # create group of population rate plots:
        if self.title is not None:
            group_name = self.title
        else:
            if use_tex:
                group_name = r'\textbf{Population Rates (Hz)}'
            else:
                group_name = 'Population Rates (Hz)'
        subplot_group = MySubplotGroup('rtm', group_name, 'time',
                                       hide_time_ax=hide_time_ax, y_label=[r''])

        for pop in self.pops:
            monitors = []
            if isinstance(network['pop_' + pop.name], NeuronGroup) and pop.asb_flag:

                colors_rtm = color_gradient(pop.plot_color, darken_color(pop.plot_color, 0.3), n_asb)
                for i in range(n_asb):
                    rtm = 'rtm_' + pop.name + '_asb_' + str(i + 1)
                    rtm_t, rtm_rate = trim_brian_monitor(network[rtm], network[rtm].rate, Hz,
                                                         test_range[0], test_range[1])
                    rtm_rate = filter_array(rtm_rate, sim_dt, self.filter_func, self.filter_width)
                    monitors += [[rtm_t, rtm_rate]]

                rtm_out = 'rtm_' + pop.name + '_out'
                if rtm_out in network:
                    if self.bg_in:
                        colors_rtm += ['black']

                        rtm_t, rtm_rate = trim_brian_monitor(network[rtm_out], network[rtm_out].rate, Hz,
                                                             test_range[0], test_range[1])
                        rtm_rate = filter_array(rtm_rate, sim_dt, self.filter_func, self.filter_width)
                        monitors += [[rtm_t, rtm_rate]]

                        subplot_group.add_subplot('rtm_' + pop.name, monitors, 'trace', colors_rtm,
                                                  settings.plot_params)
                    else:
                        rtm_t, rtm_rate = trim_brian_monitor(network[rtm_out], network[rtm_out].rate, Hz,
                                                             test_range[0], test_range[1])
                        rtm_rate = filter_array(rtm_rate, sim_dt, self.filter_func, self.filter_width)
                        monitor_out = [[rtm_t, rtm_rate]]

                        subplot_group.add_subplot('rtm_' + pop.name + '_out', monitor_out, 'trace',
                                                  ['black'], settings.plot_params)

                        subplot_group.add_subplot('rtm_' + pop.name, monitors, 'trace', colors_rtm,
                                                  settings.plot_params)

                else:
                    subplot_group.add_subplot('rtm_' + pop.name, monitors, 'trace', colors_rtm,
                                              settings.plot_params)

                if self.overall:
                    rtm = 'rtm_' + pop.name
                    rtm_t, rtm_rate = trim_brian_monitor(network[rtm], network[rtm].rate, Hz,
                                                         test_range[0], test_range[1])
                    rtm_rate = filter_array(rtm_rate, sim_dt, self.filter_func, self.filter_width)
                    subplot_group.add_subplot('rtm_' + pop.name, [[rtm_t, rtm_rate]], 'trace',
                                              ['dimgray'], settings.plot_params)

            else:
                rtm = 'rtm_' + pop.name
                rtm_t, rtm_rate = trim_brian_monitor(network[rtm], network[rtm].rate, Hz,
                                                     test_range[0], test_range[1])
                rtm_rate = filter_array(rtm_rate, sim_dt, self.filter_func, self.filter_width)
                monitors += [[rtm_t, rtm_rate]]
                subplot_group.add_subplot(rtm, monitors, 'trace', [pop.plot_color],
                                          settings.plot_params)

        if self.y_max is not None:
            for subplot, y_max_ in zip(subplot_group.subplots, self.y_max):
                _, y_ticks = get_height(y_max_)
                subplot.fix_y_axis([-0.01*y_max_, y_max_], y_ticks)

        return subplot_group


class PlotV:
    """
    object to create membrane potential plots
    """
    def __init__(self, pop, asb=[], title=None):
        """
        Args:
            pop: name of population to plot
            asb: list of assemblies to plot
        """
        self.pop = pop
        self.asb = asb
        if len(asb) > 0:
            self.monitor_types = ['v_%d' % asb_idx for asb_idx in asb]
        else:
            self.monitor_types = ['v']
        self.title = title

    def perform_plot(self, network, settings, test_range):
        use_tex = settings.options['use_tex']
        hide_time_ax = settings.plot_params['hide_time_ax'].get_param()
        plot_params = settings.plot_params

        if self.title is not None:
            group_name = self.title
        else:
            if self.pop.asb_flag:
                if use_tex:
                    group_name = r'\textbf{Membrane Potentials in %s Assemblies \#%s (mV)}' % (self.pop.name.upper(), self.asb)
                else:
                    group_name = 'Membrane Potentials in %s Assemblies %s (mV)' % (self.pop.name.upper(), self.asb)
            else:
                if use_tex:
                    group_name = r'\textbf{Membrane Potentials of %s (mV)}' % (self.pop.name.upper())
                else:
                    group_name = 'Membrane Potentials of %s (mV)' % (self.pop.name.upper())

        subplot_group = MySubplotGroup('voltage', group_name, 'time',
                                       hide_time_ax=hide_time_ax, y_label=[r''])

        e_rest = self.pop.model.e_rest.get_param() / mV
        v_reset = self.pop.model.v_reset.get_param() / mV
        v_thres = self.pop.model.v_thres.get_param() / mV

        monitors = []
        if len(self.asb) > 0:
            for asb_idx in reversed(self.asb):
                if asb_idx == 0:
                    stm_name = 'stm_%s_bg_v' % (self.pop.name)
                else:
                    stm_name = 'stm_%s_asb_%d_v' % (self.pop.name, asb_idx)
                v_time, v_val = trim_brian_monitor_group(network[stm_name], network[stm_name].v, mV,
                                                         test_range[0], test_range[1])
                # bound low values of v for density plots
                v_val[v_val < e_rest] = e_rest
                monitors += [[v_time, v_val]]
                subplot_group.add_subplot(stm_name, monitors, 'density',
                                          [self.pop.plot_color], plot_params)
        else:
            stm_name = 'stm_%s_v' % self.pop.name
            v_time, v_val = trim_brian_monitor_group(network[stm_name], network[stm_name].v, mV,
                                                     test_range[0], test_range[1])
            # bound low values of v for density plots
            v_val[v_val < e_rest] = e_rest
            monitors += [[v_time, v_val]]
            subplot_group.add_subplot(stm_name, monitors, 'density',
                                      [self.pop.plot_color], plot_params)

        for subplot_obj in subplot_group.subplots:
            subplot_obj.fix_y_axis([e_rest - np.abs(e_rest - v_thres) * 0.05, v_thres],
                                   [v_reset, v_thres],
                                   ticklabels=['-60', '-50'])

        return subplot_group


class PlotV1D:
    """
    object to create snapshots of membrane potential distribution
    """
    def __init__(self, pop, time=[], asb=[], title=None):
        """
        Args:
            pop: name of population to plot
            time: time of snapshots
            asb: list of assemblies to plot
        """
        self.pop = pop
        self.time = time
        self.asb = asb
        self.title = title
        self.monitor_types = ['v_%d' % asb_idx for asb_idx in asb]

    def perform_plot(self, network, settings, test_range):
        use_tex = settings.options['use_tex']
        plot_params = settings.plot_params

        if self.title is not None:
            group_name = self.title
        else:
            if use_tex:
                group_name = r'\textbf{Membrane Potentials}'
            else:
                group_name = 'Membrane Potentials'
        subplot_group = MySubplotGroup('v1d', group_name, 'horizontal', y_label=[r'\# Neurons'])

        if len(self.asb) > 0 or len(self.time) > 0:
            for asb_idx in self.asb:
                stm_name = 'stm_%s_asb_%d_v' % (self.pop.name, asb_idx)
                for time_i in self.time:
                    time_arg = np.argmin(np.abs(network[stm_name].t - time_i))
                    v_snapshot = np.array(network[stm_name].v[:, time_arg] / mV)

                    subplot_group.add_subplot(stm_name, [v_snapshot], 'hist',
                                              ['darkgray'], plot_params,
                                              height_ratio=1.0, x_label=r'$v$ (mV)')

            v_thres = self.pop.model.v_thres.get_param() / mV
            v_reset = self.pop.model.v_reset.get_param() / mV

            for subplot_obj in subplot_group.subplots:
                subplot_obj.fix_x_axis([v_reset, v_thres], [-60, -55, -50])

            for subplot_obj in subplot_group.subplots:
                subplot_obj.fix_y_axis([0, 150], [0, 100], ticklabels=['0', '100'])
        return subplot_group


def create_figure(settings, plot_params, n_time_ticks, subplot_groups, test_range, test_data, events,
                  t_lines=[], time_bar=None):
    """
    creates figure where predetermined subplots are displayed
    Args:
        settings: simulation settings
        plot_params: plot parameters
        n_time_ticks: number of ticks in time axis
        subplot_groups: groups of subplots (previously determined)
        test_range: time range of plots [start, stop]
        test_data: test data (previously calculated)
        events: simulation events
        t_lines: vertical lines making specific times
        time_bar: time_scale bar
    Returns:
        pyplot figure object
    """

    use_tex = settings.options['use_tex']
    if use_tex:
        rc('text', usetex=True)
    else:
        rc('text', usetex=False)

    time_stamp = settings.options['time_stamp']

    # get fig dimensions in inches
    fig_width = plot_params['fig_width'].get_param() / (cmeter * 2.54)
    fig_height = plot_params['fig_height'].get_param() / (cmeter * 2.54)
    font_size = plot_params['text_font'].get_param()

    # create figure object:
    fig = plt.figure(figsize=(fig_width, fig_height))
    # fig.canvas.set_window_title('Test ' + time_stamp)

    # get total number of columns and rows of subplots
    num_groups = len(subplot_groups)
    num_vert = 0
    num_horiz = 0
    last_vert = 0
    for group_idx in range(len(subplot_groups)):
        group = subplot_groups[group_idx]
        if group.group_type == 'time':
            last_vert = group_idx
        num_vert += group.get_num_vert()
        if group.get_num_horiz() > num_horiz:
            num_horiz = group.get_num_horiz()

    # calculate array of width ratios for all subplots:
    if num_horiz > 1:
        width_ratios = []
        size_horiz = 1/(1.25*num_horiz)
        for i in range(num_horiz):
            width_ratios += [size_horiz]
            if i < num_horiz - 1:
                width_ratios += [0.25*size_horiz]
        width_ratios = width_ratios
    else:
        width_ratios = [1]

    # calculate array of height ratios for all subplots:
    height_ratios = []
    for group_idx in range(len(subplot_groups)):
        group = subplot_groups[group_idx]
        for i in range(group.get_num_vert()):
            if group.group_type == 'time':
                height_ratios += [1.0*group.get_height_ratios(i)]
            else:
                height_ratios += [1.9*group.get_height_ratios(i)]
        if group_idx < len(subplot_groups) - 1:
            if group_idx == last_vert:
                height_ratios += [1.1]
            else:
                height_ratios += [0.5]

    # create grid where all subplots will be added:
    num_vert_plots = num_vert + num_groups - 1
    num_horiz_plots = num_horiz * 2 - 1
    gs = gridspec.GridSpec(num_vert_plots, num_horiz_plots,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios)

    # add all subplots to grid:
    idx_vert = 0
    for group_idx in range(len(subplot_groups)):
        group = subplot_groups[group_idx]
        if group.group_type == 'time':
            for i in range(len(group.subplots)):
                group.subplots[i].attr_ax(fig.add_subplot(gs[idx_vert, 0:(num_horiz*2-1)]))
                idx_vert += 1
        else:
            idx_horiz = 0
            for i in range(len(group.subplots)):
                group.subplots[i].attr_ax(fig.add_subplot(gs[idx_vert, idx_horiz]))
                idx_horiz += 2
            idx_vert += 1
        idx_vert += 1

    # initialise groups and attribute line objects to each subplot:
    for group_idx in range(len(subplot_groups)):
        group = subplot_groups[group_idx]
        if group_idx == last_vert:
            time_labels = True
        else:
            time_labels = False
        group.init_group_format(time_labels)
        for i in range(len(group.subplots)):
            group.subplots[i].add_lines()

    # draw time bar
    if time_bar is not None:
        for group_idx in range(len(subplot_groups)):
            group = subplot_groups[group_idx]
            if group.group_type == 'time':
                first_plot = group.subplots[0]

                end_time = test_range[1] / second
                start_time = (test_range[1] - time_bar) / second
                time_range = (test_range[1] - test_range[0]) / second
                start_prop = (start_time - (test_range[0] / second)) / time_range
                end_prop = 1.0

                # Determine the position of the time scale bar
                y_bott, y_top = first_plot.ax.get_ylim()
                y_pos = y_top + 0.02 * (y_top - y_bott)  # Adjust this value as needed

                # Draw the horizontal line for the time scale
                first_plot.ax.axhline(y_pos, xmin=start_prop, xmax=end_prop,
                                      linewidth=1, color='black', clip_on=False)

                # Add text annotation for the time scale
                time_scale_text = r'%.0f ms' % (time_bar  / ms)
                first_plot.ax.text((start_prop + end_prop) / 2, 1.02, time_scale_text,
                                   ha='center', va='bottom', transform=first_plot.ax.transAxes)
                break

    for group_idx in range(len(subplot_groups)):
        group = subplot_groups[group_idx]

        if group.group_type == 'time':
            group.set_time_axes(test_range[0] / second, test_range[1] / second, n_time_ticks)
            for sub_plot in group.subplots:
                sub_plot.add_vlines(t_lines)

        for sub_plot in group.subplots:
            sub_plot.set_lines()

    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    return fig
