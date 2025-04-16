from brian2 import *
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from textwrap import wrap
from imported_code.detect_peaks import detect_peaks
from my_code.aux_functions import (get_synchronicity, get_unit_rates,
                                   get_isi_cv, trim_brian_monitor, filter_array, xprint,
                                   gaussian_function, square_func, calc_low_pass_filter)
from my_code.plots import create_figure, PlotV1D
from my_code.network import VoltageStimulus, NeuronPopulationSettings, Synapses


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


class NetworkTests:
    """
    object specifying the properties of a set of tests to be performed on the network
    """
    def __init__(self,
                 monitors,
                 start,
                 stop,
                 max_record=np.inf,
                 test_list=[],
                 plot_list=[],
                 n_time_ticks=5,
                 time_bar=None):
        """
        Args:
            monitors: recording monitors needed for tests
            start: start time of tests
            stop: stop time of tests
            max_record: maximum number of neurons to record
            test_list: list of tests to be performed
            plot_list: list of plots to be created
            n_time_ticks: number of ticks on the time axis of plots
        """
        self.start = start
        self.stop = stop
        self.max_record = max_record
        self.test_list = test_list
        self.plot_list = plot_list
        self.n_time_ticks = n_time_ticks
        self.subplot_groups = []
        self.test_mons = []
        self.time_bar = time_bar

        # identify necessary recording monitors for each test and plot
        for test_plot_obj in (test_list + plot_list):
            for mon_type in test_plot_obj.monitor_types:
                if hasattr(test_plot_obj, 'pops'):
                    for pop in test_plot_obj.pops:
                        if [pop, mon_type] not in self.test_mons:
                            self.test_mons.append([pop, mon_type])
                if hasattr(test_plot_obj, 'pop'):
                    if [test_plot_obj.pop, mon_type] not in self.test_mons:
                        self.test_mons.append([test_plot_obj.pop, mon_type])
                if hasattr(test_plot_obj, 'syn'):
                    if [test_plot_obj.syn, mon_type] not in self.test_mons:
                        self.test_mons.append([test_plot_obj.syn, mon_type])

        # check if monitors overlap and create all the necessary MonitorSettings objects
        for mon_inst in self.test_mons:
            pop, mon_type = mon_inst

            match_monitor = False

            # check if it overlaps with any other monitors
            for monitor in monitors:
                if (monitor.pop == pop) and (monitor.mon_type == mon_type):

                    if ((self.start <= monitor.stop_record <= self.stop) or
                            (self.start <= monitor.start_record <= self.stop)):

                        match_monitor = True

                        if self.start < monitor.start_record:
                            monitor.start_record = self.start
                        if self.stop > monitor.stop_record:
                            monitor.stop_record = self.stop

            if not match_monitor:
                monitors += [MonitorSettings(pop, mon_type, self.start, self.stop, self.max_record)]

    def perform_tests(self, network, settings, test_range, test_data, log=None):
        for test_func in self.test_list:
            test_func.perform_test(network, settings, test_range, test_data, log=log)

    def create_plots(self, network, settings, events, test_range, test_data):
        if len(self.plot_list) > 0:
            vlines = []
            for plot_func in self.plot_list:
                self.subplot_groups.append(plot_func.perform_plot(network, settings, test_range))
                if isinstance(plot_func, PlotV1D):
                    vlines = plot_func.time / second

            plot_params = settings.plot_params
            options = settings.options

            fig_to_save = create_figure(settings, plot_params, self.n_time_ticks,
                                        self.subplot_groups, test_range, test_data, events,
                                        t_lines=vlines, time_bar=self.time_bar)

            fig_name = (options['output_dir'] + options['time_stamp'] + '/sim' +
                        str(options['sim_idx']) + '_[%.3f-%.3f]s' %
                        (test_range[0] / second, test_range[1] / second))
            fig_to_save.savefig(fig_name + '.png', dpi=300, bbox_inches='tight')
            fig_to_save.savefig(fig_name + '.svg', dpi=600, bbox_inches='tight')
            plt.close(fig_to_save)


class MonitorSettings:
    """
    object specifying a network monitor needed by the simulation
    """
    def __init__(self,
                 pop,
                 mon_type,
                 start_record,
                 stop_record,
                 max_record=np.inf):
        """
        Args:
            pop: name of neuron population
            mon_type: type of monitor needed ['spike', 'v', 'curr']
            start_record: recording start time
            stop_record: recording stop time
            max_record: maximum number of neurons to record
        """
        self.pop = pop
        self.mon_type = mon_type
        self.start_record = start_record
        self.stop_record = stop_record
        self.max_record = max_record

        self.created_monitors = []

    def create_monitors(self, network, settings, log=None):

        if self.mon_type == 'spike':
            self.pop.create_firing_monitors(self.created_monitors, network, settings, self.max_record)

        if self.mon_type == 'v':
            self.pop.create_v_monitors(self.created_monitors, network, settings, self.max_record)

        if 'v_' in self.mon_type:
            asb_idx = int(self.mon_type[2:])
            self.pop.create_single_v_monitor(self.created_monitors, network, settings, asb_idx, self.max_record)

        if self.mon_type == 'curr_all':
            self.pop.create_all_current_monitors(self.created_monitors, network, settings, self.max_record)

        if 'curr_' in self.mon_type:
            pre_pop = self.mon_type[-1]
            self.pop.create_single_current_monitor(self.created_monitors, network, settings, pre_pop, self.max_record)

        if 'weight_' in self.mon_type:
            pre_pop = self.mon_type[-1]
            self.pop.create_weight_monitors(self.created_monitors, network, settings, pre_pop, self.max_record)

        if self.mon_type == 'syn_depr':
            self.pop.create_syn_depr_monitors(self.created_monitors, network, settings, self.max_record)

        if len(self.created_monitors) > 0:
            xprint('%.4f s: Created monitors %s' % (network.t / second, self.created_monitors), log)

    def delete_monitors(self, network, running_monitor_groups, log=None):

        deleted_monitors = []
        for mon in self.created_monitors:
            del_mon = True
            for mon_group in running_monitor_groups:
                if ((mon_group.stop_record > self.stop_record) and
                        (mon in mon_group.created_monitors)):
                        del_mon = False

            if del_mon and mon in network:
                del network[mon]
                deleted_monitors += [mon]

        if len(self.created_monitors) > 0:
            xprint('%.4f s: Deleted monitors %s' % (network.t / second, deleted_monitors), log)


class TestFiring:
    """
    object to test the following firing properties:
    - unit firing rates
    - regularity of unit firing (ISI CV)
    - network synchronicity
    """
    def __init__(self, pops=[], max_psd_freq=300*Hz, store_group=False):
        """
        Args:
            pops: list of populations to test
            max_psd_freq: maximum frequency of PSD calculations
        """
        self.pops = pops
        self.max_psd_freq = max_psd_freq
        self.monitor_types = ['spike']
        self.store_group = store_group

    def perform_test(self, network, settings, test_range, test_data, log=log):
        n_asb = settings.net_objects.n_asb

        sim_dt = defaultclock.dt

        xprint('\n============== UNIT FIRING RATES ===============', log)

        for pop in self.pops:
            if pop.asb_flag:
                spm_str = 'spm_' + pop.name + '_out'
                if spm_str in network:
                    get_unit_rates(test_data, network[spm_str], test_range, pop.name + '_out', log=log)

                for asb_idx in range(n_asb):
                    spm_str = 'spm_' + pop.name + '_asb_' + str(asb_idx + 1)
                    if spm_str in network:
                        get_unit_rates(test_data, network[spm_str], test_range,
                                       pop.name + '_asb_' + str(asb_idx + 1), log=log)
            else:
                spm_str = 'spm_' + pop.name
                if spm_str in network:
                    get_unit_rates(test_data, network[spm_str], test_range, pop.name, log=log)

        xprint('\n========== REGULARITY OF UNIT FIRING ===========', log)

        for pop in self.pops:
            if pop.asb_flag:
                spm_str = 'spm_' + pop.name + '_out'
                if spm_str in network:
                    get_isi_cv(test_data, network[spm_str], test_range, pop.name + '_out', log=log)

                for asb_idx in range(n_asb):
                    spm_str = 'spm_' + pop.name + '_asb_' + str(asb_idx + 1)
                    if spm_str in network:
                        get_isi_cv(test_data, network[spm_str], test_range,
                                   pop.name + '_asb_' + str(asb_idx + 1), log=log)

            else:
                spm_str = 'spm_' + pop.name
                if spm_str in network:
                    get_isi_cv(test_data, network[spm_str], test_range, pop.name, log=log)

        xprint('\n================ SYNCHRONICITY =================', log)

        for pop in self.pops:
            if pop.asb_flag:
                spm_str = 'spm_' + pop.name + '_out'
                if spm_str in network:
                    get_synchronicity(test_data, network[spm_str], test_range, pop.name + '_out',
                                      self.max_psd_freq, log=log)

                for asb_idx in range(n_asb):
                    spm_str = 'spm_' + pop.name + '_asb_' + str(asb_idx + 1)
                    if spm_str in network:
                        get_synchronicity(test_data, network[spm_str], test_range,
                                          pop.name + '_asb_' + str(asb_idx + 1),
                                          self.max_psd_freq, log=log)

            else:
                spm_str = 'spm_' + pop.name
                if spm_str in network:
                    get_synchronicity(test_data, network[spm_str], test_range, pop.name,
                                      self.max_psd_freq, log=log)

        if self.store_group:
            options = settings.options
            sim_params = settings.sim_params
            group_params = options['group_param_array'].keys()

            # create replay properties file
            firing_file_path = options['output_dir'] + options['time_stamp'] + \
                               '/0_group_firing_[%.3f-%.3f]s.txt' % (test_range[0] / second, test_range[1] / second)
            if not os.path.exists(firing_file_path):
                with open(firing_file_path, 'w') as firing_file:

                    file_header = 'sim# \t '
                    for param_name in group_params:
                        file_header += param_name + ' \t '

                    for pop_sett in self.pops:
                        file_header += 'mean_rate_' + pop_sett.name + '_asb%d \t ' % settings.net_objects.n_asb
                    firing_file.write(file_header + '\n')
                    firing_file.close()

            param_vals = '%d \t ' % options['sim_idx']
            for param_name in group_params:
                param_vals += sim_params[param_name].get_str() + ' \t '

            line_str = param_vals
            for pop_sett in self.pops:
                line_str += ' \t %.3f' % (test_data['mean_rate_' + pop_sett.name + '_asb_%d' % settings.net_objects.n_asb])

            firing_file = open(firing_file_path, 'a')
            firing_file.write(line_str + '\n')
            firing_file.close()


class TestFitV:
    """
    object to test if v distribution can be fitted by a function
    """
    def __init__(self, pops=[], asb=[], time=[]):
        """
        Args:
            pops: list of populations to test
        """
        self.pops = pops
        self.asb = asb
        self.time = time
        self.monitor_types = ['v']

    def perform_test(self, network, settings, test_range, test_data, log=log):

        sim_params = settings.sim_params
        events = settings.events
        options = settings.options
        group_params = options['group_param_array'].keys()

        for pop in self.pops:
            for asb_idx in self.asb:
                stm_name = 'stm_%s_asb_%d_v' % (pop.name, asb_idx)
                for time_i in self.time:

                    # create replay properties file
                    v_file_path = options['output_dir'] + options['time_stamp'] + \
                                  '/0_group_v_pop_%s_%d_%.3fs.txt' % (pop.name, asb_idx, time_i)
                    if not os.path.exists(v_file_path):
                        with open(v_file_path, 'w') as v_file:

                            v_header = 'sim# \t '
                            for param_name in group_params:
                                v_header += param_name + ' \t '

                            v_header += ('v_mean \t v_std \t gauss_mean \t gauss_std \t '
                                         'square1_w \t square1_d \t square2_w')
                            v_file.write(v_header + '\n')
                            v_file.close()

                    param_vals = '%d \t ' % options['sim_idx']
                    for param_name in group_params:
                        param_vals += sim_params[param_name].get_str() + ' \t '

                    thres = pop.model.v_thres.get_param() / mV
                    time_arg = np.argmin(np.abs(network[stm_name].t - time_i))
                    v_snapshot = np.array(network[stm_name].v[:, time_arg] / mV)
                    counts, bin_edges = np.histogram(v_snapshot, bins=40, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_width = bin_edges[1] - bin_edges[0]

                    gauss_mean = np.nan
                    gauss_width = np.nan
                    square1_width = np.nan
                    square1_delta = np.nan
                    square2_width = np.nan

                    v_mean = np.mean(v_snapshot)
                    v_std = np.std(v_snapshot)


                    fit_gauss = False
                    fit_thres = -59
                    v_tofit = v_snapshot[v_snapshot > fit_thres]
                    # fit to gaussian
                    try:
                        fit_gauss_params, _ = curve_fit(gaussian_function, bin_centers, counts,
                                                  p0=[1, np.mean(v_tofit), np.std(v_tofit)])
                        fit_gauss = True
                        gauss_mean, gauss_width = fit_gauss_params[1:]

                        xprint('\n=========== V GAUSSIAN FIT ===========', log)
                        xprint('v (mean +- std) = (%.3f +- %.3f) mV' % (v_mean, v_std), log)
                        xprint('gauss mean = %.3f mV' % gauss_mean, log)
                        xprint('gauss sigma = %.3f mV' % gauss_width, log)

                        # find r2 of gaussian
                        residuals = counts - gaussian_function(bin_centers, *fit_gauss_params)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
                        gauss_r2 = 1 - (ss_res / ss_tot)
                        xprint('gauss r2 = %f\n' % gauss_r2, log)

                        # # gauss square
                        # square1_mean = gauss_mean
                        # square1_width = 3.07 * gauss_width
                        # square1_delta = thres - (square1_mean + square1_width / 2)
                        # xprint('square mean = %.3f mV' % square1_mean, log)
                        # xprint('square width = %.3f mV' % square1_width, log)
                        # xprint('square delta = %.3f mV' % square1_delta, log)
                        # # find r2 of gauss_square
                        # residuals = counts - square_func(bin_centers, square1_mean, square1_width)
                        # ss_res = np.sum(residuals ** 2)
                        # ss_tot = np.sum((counts - np.mean(counts)) ** 2)
                        # square1_r2 = 1 - (ss_res / ss_tot)
                        # xprint('gauss square r2 = %f' % square1_r2, log)

                    except RuntimeError:
                        xprint("WARNING: Couldn't fit voltage distribution to gaussian", log)

                    # print to file
                    v_str = param_vals + '%.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f' % (
                        v_mean, v_std, gauss_mean, gauss_width, square1_width, square1_delta, square2_width)
                    v_file = open(v_file_path, 'a')
                    v_file.write(v_str + '\n')
                    v_file.close()

                    # plot
                    fig, ax = plt.subplots(figsize=(5 / 2.54, 2 / 2.54))
                    font_size = 9
                    counts_plot, bin_edges_plot = np.histogram(v_snapshot[v_snapshot > fit_thres], density=True)
                    bin_centers_plot = (bin_edges_plot[:-1] + bin_edges_plot[1:]) / 2

                    ax.bar(bin_centers_plot, counts_plot, width=0.25, color='black')
                    # ax.scatter(bin_centers_fit, counts_fit, s=1.0, color='black')

                    v_range = max(v_tofit) - min(v_tofit)
                    x_values = np.linspace(min(bin_centers_plot) - v_range * 0.05, max(bin_centers_plot) + v_range * 0.05, 100)
                    # ax.text(0.02, 0.95, r'$<v>$ = (%.3f +- %.3f) mV' %
                    #         (v_mean, v_std), transform=ax.transAxes, fontsize=font_size)
                    if fit_gauss:
                        # ax.text(0.02, 0.90, r'Gauss $R^2$ = %.3f' % gauss_r2, transform=ax.transAxes, fontsize=font_size)
                        # ax.text(0.02, 0.85, r'- $\mu$ = %.3f mV' % gauss_mean, transform=ax.transAxes, fontsize=font_size)
                        # ax.text(0.02, 0.80, r'- $\sigma$ = %.3f mV' % gauss_width, transform=ax.transAxes, fontsize=font_size)
                        ax.plot(x_values, gaussian_function(x_values, *fit_gauss_params), color='black', lw=2)
                        ax.plot(x_values, gaussian_function(x_values, *fit_gauss_params), color='darkgray', lw=1.5, alpha=0.95)


                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(1.5)
                    ax.set_ylim([0, 1])
                    ax.set_yticks([])
                    ax.set_xlim([-56.2, -49.8])
                    ax.axvline(-50, color='black', lw=1.5, ls='--')
                    ax.set_xticks([-55, -50], labels=['', ''], fontsize=font_size)
                    # ax.set_xlabel(r'$v$ (mV)', fontsize=font_size)
                    # ax.set_ylabel('density', fontsize=font_size)

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(direction='inout', length=6, width=1.5, colors='black', labelsize=font_size)
                    ax.set_axisbelow(False)  # <-- added

                    options = settings.options
                    fig_name = (options['output_dir'] + options['time_stamp'] + '/sim' +
                                str(options['sim_idx']) + '_v_pop_%s_asb_%s_%.3fs' % (pop.name, asb_idx, time_i))
                    fig.savefig(fig_name + '.png', dpi=300, bbox_inches='tight')
                    fig.savefig(fig_name + '.svg', bbox_inches='tight')


class TestReplay:
    """
    object to test if an attempted sequence replay has succeeded
    """
    def __init__(self, pop, filter_width=2*ms, min_height=30*Hz, min_dist=1.0*ms, detect_range=80*ms,
                 act_lim=0.0, loop_times=1):
        """
        Args:
            pop: name of population to test
            filter_width: gaussian width to filter population rate
            detect_range: time range of peak detection around stimulation time
        """
        self.pop = pop
        self.filter_width = filter_width
        self.min_height = min_height
        self.min_dist = min_dist
        self.detect_range = detect_range
        self.act_lim = act_lim
        self.loop_times = loop_times
        self.monitor_types = ['spike']


    def perform_test(self, network, settings, test_range, test_data, log=log):

        sim_params = settings.sim_params
        events = settings.events
        options = settings.options
        group_params = options['group_param_array'].keys()

        # create replay properties file
        replay_file_path = options['output_dir'] + options['time_stamp'] + '/0_group_replay.txt'
        if not os.path.exists(replay_file_path):
            with open(replay_file_path, 'w') as replay_file:

                replay_header = 'sim# \t '
                for param_name in group_params:
                    replay_header += param_name + ' \t '

                replay_header += 'stim# \t replay \t avg_speed \t asy_speed \t asy_act \t asy_width \t explosion'
                replay_file.write(replay_header + '\n')
                replay_file.close()

        # create replay trajectories file
        traject_file_path = options['output_dir'] + options['time_stamp'] + '/0_group_trajects.txt'
        if not os.path.exists(traject_file_path):
            with open(traject_file_path, 'w') as traject_file:
                trajects_header = 'sim# \t '
                for param_name in group_params:
                    trajects_header += param_name + ' \t '

                trajects_header += 'stim# \t asb \t act \t width \t replay'
                traject_file.write(trajects_header + '\n')
                traject_file.close()

        asb_size = self.pop.asb_size

        sim_dt = defaultclock.dt
        test_start = test_range[0]
        test_stop = test_range[1]

        stim_events = []
        for event in events:
            if isinstance(event, VoltageStimulus):
                stim_events += [event]

        stim_start = []
        stim_stop = []
        stim_unchecked = []
        for event in stim_events:
            if isinstance(event, VoltageStimulus):
                if len(np.array(list(event.jump_dict.keys()))) > 0:
                    stim_start += [np.min(np.array(list(event.jump_dict.keys()))) * second]
                    stim_stop += [np.max(np.array(list(event.jump_dict.keys()))) * second]
                else:
                    stim_start += [event.time]
                    stim_stop += [event.time]
                stim_unchecked += [not event.checked_replay]

        # select stimuli that are unchecked and within calc_range:
        stim_start = np.array(stim_start / second) * second
        stim_stop = np.array(stim_stop / second) * second
        stim_unchecked = np.array(stim_unchecked)
        stim_check = (stim_start > test_start) * (stim_start < test_stop) * \
                     (stim_stop > test_start) * (stim_stop < test_stop) * stim_unchecked

        stim_idx = np.argwhere(stim_check)[0][0]
        stim_events[stim_idx].checked_replay = True

        spm_asb_1_time = network['spm_' + self.pop.name + '_asb_1'].t
        spm_asb_1_i = network['spm_' + self.pop.name + '_asb_1'].i
        n_p_asb = self.pop.asb_size

        # neurons that spiked during stim_time
        asb1_spiked, asb1_counts = np.unique(
            spm_asb_1_i[(spm_asb_1_time >= stim_start[stim_idx]) &
                        (spm_asb_1_time <= stim_stop[stim_idx])],
            return_counts=True)
        stim_str = 'stim_time [%.4f:%.4f] s: %d / %d neurons spiked (%.1f%%)' % \
                   (stim_start[stim_idx], stim_stop[stim_idx],
                    len(asb1_spiked), n_p_asb, 100 * len(asb1_spiked) / n_p_asb)
        if len(asb1_counts) > 0:
            stim_str += '\t %d more than once' % np.sum(asb1_counts >= 2)
            stim_str += ' (max fired %d times)' % np.max(asb1_counts)
        xprint(stim_str, log)

        n_asb = settings.net_objects.n_asb

        check_replay = True
        check_explosion = False
        replay_spikes = np.full(n_asb, np.nan)
        replay_width = np.full(n_asb, np.nan)

        prev_peak = 0 * second

        test_data['detect_min_array'] = np.array([])
        test_data['detect_max_array'] = np.array([])
        test_data['spike_count_min_array'] = np.array([])
        test_data['spike_count_max_array'] = np.array([])
        checked_peak_times = np.array([])

        # if it loops check twice
        if self.loop_times > 1:
            asb_indices = np.tile(np.arange(0, n_asb), self.loop_times)
        else:
            asb_indices = np.arange(0, n_asb)

        checked_stim = False
        for i in asb_indices:
            # detection range for 1st asb:
            if i == 0 and not checked_stim:
                detect_min = stim_start[stim_idx] - self.detect_range / 4
                detect_max = stim_stop[stim_idx] + self.detect_range * 3 / 4
                checked_stim = True

            # detection range for subsequent assemblies:
            else:
                detect_min = prev_peak - self.detect_range / 4
                detect_max = prev_peak + self.detect_range * 3 / 4

            # store for plotting
            test_data['detect_min_array'] = np.append(test_data['detect_min_array'], detect_min)
            test_data['detect_max_array'] = np.append(test_data['detect_max_array'], detect_max)

            # find peaks
            rtm_asb_mon = network['rtm_' + self.pop.name + '_asb_' + str(i + 1)]
            rtm_asb_t, rtm_asb_r = trim_brian_monitor(rtm_asb_mon, rtm_asb_mon.rate, Hz,
                                                      test_range[0], test_range[1])
            rtm_asb_r = filter_array(rtm_asb_r / Hz, sim_dt / second,
                                     'gauss', self.filter_width / second) * Hz

            peak_idx = detect_peaks(rtm_asb_r, mph=self.min_height, mpd=int(self.min_dist / sim_dt))

            peak_times = rtm_asb_t[peak_idx] * second
            peak_heights = rtm_asb_r[peak_idx]

            xprint('asb %d:' % (i + 1), log)

            xprint('\t all peaks: %s : %s' % (peak_times, peak_heights), log)

            # find which of the detected peaks corresponds to evoked replay:
            evk_peak_idx = (peak_times >= detect_min) & \
                           (peak_times <= detect_max)

            xprint('\t replay peak: %s : %s' % (peak_times[evk_peak_idx], peak_heights[evk_peak_idx]), log)

            # only one 'evoked peak' should be detected within the detection range
            if np.sum(evk_peak_idx) == 0:
                xprint('\t Replay FAILED on asb %d: no peaks detected within detection range' % (i + 1), log)
                check_replay = False
                prev_peak = 0 * second

            elif np.sum(evk_peak_idx) == 1:
                xprint('\t delay: %s' % (peak_times[evk_peak_idx] - prev_peak), log)

                if peak_times[evk_peak_idx] - prev_peak < self.min_dist:
                    xprint('\t Replay FAILED on asb %d: delay is too small! (< %.2f ms)' % ((i + 1), self.min_dist / ms),
                           log)
                    check_replay = False

                prev_peak = peak_times[evk_peak_idx]
                checked_peak_times = np.append(checked_peak_times, peak_times[evk_peak_idx])

                # if peak_heights[evk_peak_idx] > 1000 * Hz:
                #     xprint('Replay FAILED on asb %d: detected peak is higher than threshold' % (i + 1), log)
                #     check_explosion = True
                #     check_replay = False
            else:
                xprint('\t Replay FAILED on asb %d: too many peaks detected' % (i + 1), log)
                check_replay = False
                # could be that there are several low peaks if activity is dying out.
                # to check for explosion, check if the height of any peak is above threshold:
                if (peak_heights > self.min_height).any():
                    check_explosion = True
                    xprint('\t Activity EXPLOSION detected!' % (i + 1), log)
                prev_peak = 0 * second

            if not check_replay:
                break

            # Fit Gaussian
            gauss_time = rtm_asb_t[(rtm_asb_t >= detect_min / second) & (rtm_asb_t <= detect_max / second)]
            gauss_rate = rtm_asb_r[(rtm_asb_t >= detect_min / second) & (rtm_asb_t <= detect_max / second)] / Hz

            gauss_height = np.max(gauss_rate)
            gauss_mean = gauss_time[np.argmax(gauss_rate)]

            fit_gauss = False
            gauss_sigma = np.nan
            gauss_r2 = np.nan
            try:
                popt, _ = curve_fit(lambda t, sigma: gaussian(t, gauss_height, gauss_mean, sigma),
                                    gauss_time, gauss_rate, p0=[0.001], bounds=(0, np.inf))
                gauss_sigma = popt[0]
                residuals = gauss_rate - gaussian(gauss_time, gauss_height, gauss_mean, gauss_sigma)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((gauss_rate - np.mean(gauss_rate)) ** 2)
                gauss_r2 = 1 - (ss_res / ss_tot)
            except RuntimeError:
                xprint("\t WARNING: Couldn't fit voltage distribution to gaussian", log)

            # check if fit is good
            if (gauss_r2 > 0.5) and ~np.isnan(gauss_r2):
                fit_gauss = True

            if fit_gauss:
                # Calculate FWHM
                fwhm = 2 * np.sqrt(2 * np.log(2)) * np.abs(gauss_sigma) * 1e3  # from second to ms
                replay_width[i] = fwhm
                xprint('\t Fitted Gaussian with FWHM = %.1f ms (R2=%.1f)' % (fwhm, gauss_r2), log)

                spike_detect_max = (gauss_mean + gauss_sigma * 3) * second
                spike_detect_min = (gauss_mean - gauss_sigma * 3) * second

                # store for plotting
                test_data['spike_count_min_array'] = np.append(test_data['spike_count_min_array'], spike_detect_min)
                test_data['spike_count_max_array'] = np.append(test_data['spike_count_max_array'], spike_detect_max)

                # how many asb neurons spiked within 3 STDs
                spm_asb_time = network['spm_' + self.pop.name + '_asb_' + str(i + 1)].t
                spm_asb_i = network['spm_' + self.pop.name + '_asb_' + str(i + 1)].i
                spm_asb_i_trim = np.array(spm_asb_i)[(spm_asb_time >= spike_detect_min) &
                                                     (spm_asb_time <= spike_detect_max)]
                asb_spiked, asb_counts = np.unique(spm_asb_i_trim, return_counts=True)
                replay_spikes[i] = np.sum(asb_counts)
                xprint('\t [%.4f:%.4f] s (within +-3 Gaussian STDs):' %
                       (spike_detect_min / second, spike_detect_max / second), log)
                xprint('\t\t %d total spikes' % (replay_spikes[i]), log)
                xprint('\t\t %d / %d neurons spiked (%.1f%%) ' % \
                          (len(asb_spiked), n_p_asb, 100 * len(asb_spiked) / n_p_asb), log)
                if len(asb_counts) > 0:
                    asb_str = '\t\t %d more than once' % np.sum(asb_counts >= 2)
                    asb_str += ' (max fired %d times)' % np.max(asb_counts)
                    xprint(asb_str, log)


            else:
                check_replay = False
                xprint('\t Replay FAILED on asb %d: could not fit gaussian' % (i + 1), log)
                break


        # check dummy group remains inactive between stim and last asb peak:
        if check_replay:
            detect_min = stim_start[stim_idx]
            detect_max = prev_peak

            if 'rtm_' + self.pop.name + '_out' in network:
                rtm_out_mon = network['rtm_' + self.pop.name + '_out']
                rtm_out_t, rtm_out_r = trim_brian_monitor(rtm_out_mon, rtm_out_mon.rate, Hz,
                                                          test_range[0], test_range[1])
                rtm_out_t = rtm_out_t * second
                rtm_out_r = filter_array(rtm_out_r, sim_dt / second,
                                         'gauss', self.filter_width / second) * Hz

                rtm_out_check = rtm_out_r[(rtm_out_t >= detect_min) & (rtm_out_t <= detect_max)]
                if (rtm_out_check > 10 * Hz).any():
                    check_replay = False
                    check_explosion = True
                    xprint('Replay FAILED: dummy group exceeded threshold activity', log)
                else:
                    xprint('Dummy group remained below threshold activity', log)

        xprint('Pulse activities:', log)
        xprint('%s' % (replay_spikes[:]), log)

        xprint('Pulse FWHM [ms]:', log)
        xprint('%s' % (replay_width[:]), log)

        """ SAVE RESULTS in GROUP_REPLAY FILE """

        param_vals = '%d \t ' % options['sim_idx']
        for param_name in group_params:
            param_vals += sim_params[param_name].get_str() + ' \t '

        param_vals += str(stim_events[stim_idx]) + ' \t '

        replay_avg_speed = np.nan
        replay_asy_speed = np.nan
        replay_asy_act = np.nan
        replay_asy_width = np.nan
        replay_str = False
        if check_replay:
            xprint('Replay check SUCCEEDED!', log)
            replay_avg_speed = (checked_peak_times[-1] - checked_peak_times[0]) / (n_asb - 1)
            if n_asb >= 3:
                replay_asy_speed = (checked_peak_times[-1] - checked_peak_times[-4]) / 3
                replay_asy_act = np.mean(replay_spikes[-3:]) / asb_size
                replay_asy_width = np.mean(replay_width[-3:])

            xprint('with average speed %.1f ms / asb' % (replay_avg_speed * 1000), log)
            xprint('with asymptotic speed %.1f ms / asb' % (replay_asy_speed * 1000), log)
            xprint('with asymptotic activity of %.2f' % (replay_asy_act), log)
            xprint('with asymptotic FWHM %.1f ms' % (replay_asy_width), log)

            if replay_asy_act >= self.act_lim:
                replay_str = True
            else:
                xprint('asymptotic activity too low, replay FAILED!', log)

        else:
            xprint('Replay FAILED!', log)

        replay_str = (param_vals + '%s \t %.1f \t %.1f \t %.2f \t %.1f \t %s' %
                      (replay_str, replay_avg_speed * 1000, replay_asy_speed * 1000, replay_asy_act,
                       replay_asy_width, check_explosion))

        replay_file = open(options['output_dir'] + options['time_stamp'] + '/0_group_replay.txt', 'a')
        replay_file.write(replay_str + '\n')
        replay_file.close()

        trajects_file = open(options['output_dir'] + options['time_stamp'] + '/0_group_trajects.txt', 'a')
        for asb_idx in range(n_asb):
            trajects_str = param_vals + '%d \t %.0f \t %.1f \t %s' % (asb_idx + 1, replay_spikes[asb_idx],
                                                                    replay_width[asb_idx], check_replay)
            trajects_file.write(trajects_str + '\n')
        trajects_file.close()
