"""
tests.py
Test and monitor utilities for network simulation analysis.

This module provides classes and helpers to define, run, and plot network tests
in Brian2-based simulations. It includes:
- `NetworkTests`: orchestrates test and plot execution, manages required monitors
- `MonitorSettings`: describes and manages the lifecycle of a network monitor
- `TestReplay`: implements replay detection and analysis for sequence replay tasks
"""

from brian2 import second, Hz, defaultclock, mV, ms
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from imported_code.detect_peaks import detect_peaks
from general_code.aux_functions import trim_brian_monitor, xprint
from spiking_networks.plot_spiking_trace import create_figure, PlotV1D
from spiking_networks.network import TriggerSpikes


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))


class NetworkTests:
    """Orchestrates a set of tests and plots for a network simulation.

    Collects test and plot objects, determines required monitors, and provides
    methods to execute tests and generate plots. Used as a central test harness
    for simulation runs.
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
        """Run all tests in `self.test_list` on the network."""
        for test_func in self.test_list:
            test_func.perform_test(network, settings, test_range, test_data, log=log)

    def create_plots(self, network, settings, test_range):
        """Generate and save all plots in `self.plot_list` for the test run."""
        if len(self.plot_list) > 0:
            vlines = []
            for plot_func in self.plot_list:
                self.subplot_groups.append(plot_func.perform_plot(network, settings, test_range))

                # add vertical lines marking the times for membrane potential snapshots
                if isinstance(plot_func, PlotV1D):
                    vlines = plot_func.time / second

            plot_params = settings.plot_params
            options = settings.options

            fig_to_save = create_figure(plot_params, self.n_time_ticks,
                                        self.subplot_groups, test_range,
                                        t_lines=vlines, time_bar=self.time_bar)

            fig_name = (options['output_dir'] + options['group_label'] + '/sim' +
                        str(options['sim_idx']) + '_[%.3f-%.3f]s' %
                        (test_range[0] / second, test_range[1] / second))
            fig_to_save.savefig(fig_name + '.png', dpi=300, bbox_inches='tight')
            fig_to_save.savefig(fig_name + '.svg', dpi=600, bbox_inches='tight')
            plt.close(fig_to_save)


class MonitorSettings:
    """Describes and manages a network monitor for simulation recording.

    Stores the population, monitor type, recording window, and maximum number
    of neurons to record. Provides methods to create and delete monitors in the
    Brian2 network object.
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
            mon_type: type of monitor needed ['spike', 'v_*']
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
        """Create the required monitors in the Brian2 network object."""
        if self.mon_type == 'spike':
            self.pop.create_firing_monitors(self.created_monitors, network, settings, self.max_record)

        if 'v_' in self.mon_type:
            asb_idx = int(self.mon_type[2:])
            self.pop.create_v_monitor(self.created_monitors, network, settings, asb_idx, self.max_record)

        if len(self.created_monitors) > 0:
            xprint('%.4f s: Created monitors %s' % (network.t / second, self.created_monitors), log)

    def delete_monitors(self, network, running_monitor_groups, log=None):
        """Delete monitors from the network if no longer needed."""
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


class TestReplay:
    """Test object for detecting and analyzing sequence replay events.

    Implements logic to detect replay events, fit Gaussian curves to activity,
    and check for replay success, speed, width, and possible network explosions.
    """
    def __init__(self, pop, filter_width=2*ms, min_height=30*Hz, detect_range=80*ms, min_dist=1.0*ms):
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
        self.monitor_types = ['spike']


    def perform_test(self, network, settings, test_range, test_data, log=None):
        """Run the replay test and record results."""

        sim_params = settings.sim_params
        events = settings.events
        options = settings.options
        group_params = options['group_param_overrides'].keys()

        # create replay properties file
        replay_file_path = options['output_dir'] + options['group_label'] + '/0_group_replay.txt'
        if not os.path.exists(replay_file_path):
            with open(replay_file_path, 'w') as replay_file:

                replay_header = 'sim# \t '
                for param_name in group_params:
                    replay_header += param_name + ' \t '

                replay_header += 'stim# \t replay \t asy_speed \t asy_act \t asy_width \t explosion'
                replay_file.write(replay_header + '\n')
                replay_file.close()

        asb_size = self.pop.asb_size

        sim_dt = defaultclock.dt
        test_start = test_range[0]
        test_stop = test_range[1]

        stim_events = []
        for event in events:
            if isinstance(event, TriggerSpikes):
                stim_events += [event]

        stim_start = []
        stim_stop = []
        stim_unchecked = []
        for event in stim_events:
            if isinstance(event, TriggerSpikes):
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

            # store for plotting if needed
            test_data['detect_min_array'] = np.append(test_data['detect_min_array'], detect_min)
            test_data['detect_max_array'] = np.append(test_data['detect_max_array'], detect_max)

            # find peaks
            rtm_asb_mon = network['rtm_' + self.pop.name + '_asb_' + str(i + 1)]
            rtm_asb_t, rtm_asb_r = trim_brian_monitor(rtm_asb_mon, rtm_asb_mon.rate, Hz,
                                                      test_range[0], test_range[1])
            rtm_asb_r = gaussian_filter1d(rtm_asb_r / Hz, self.filter_width / sim_dt) * Hz

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
                prev_peak_height = 0 * Hz

            elif np.sum(evk_peak_idx) == 1:
                xprint('\t delay: %s' % (peak_times[evk_peak_idx] - prev_peak), log)

                if peak_times[evk_peak_idx] - prev_peak < self.min_dist:
                    xprint('\t Replay FAILED on asb %d: delay is too small! (< %.2f ms)' % ((i + 1), self.min_dist / ms),
                           log)
                    check_replay = False
                    # to check for explosion, check if the height of any peak is above threshold:
                    if (peak_heights > self.min_height).any():
                        check_explosion = True
                        xprint('\t Activity EXPLOSION detected!' % (i + 1), log)

                prev_peak = peak_times[evk_peak_idx]
                prev_peak_height = peak_heights[evk_peak_idx]
                checked_peak_times = np.append(checked_peak_times, peak_times[evk_peak_idx])

            else:
                xprint('\t Replay FAILED on asb %d: too many peaks detected' % (i + 1), log)
                check_replay = False

                # Could be that there are several low peaks if activity is dying out.
                # To check for explosion, check if the summed height of both peaks
                # is at least 70% of the previous peak:
                if np.sum(peak_heights) >= 0.70 * prev_peak_height:
                    check_explosion = True
                    xprint('\t Activity EXPLOSION detected!' % (i + 1), log)
                
                prev_peak = 0 * second
                prev_peak_height = 0 * Hz

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

                # store for plotting if needed
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


        # check background group remains inactive between stim and last asb peak:
        if check_replay:
            detect_min = stim_start[stim_idx]
            detect_max = prev_peak

            if 'rtm_' + self.pop.name + '_out' in network:
                rtm_out_mon = network['rtm_' + self.pop.name + '_out']
                rtm_out_t, rtm_out_r = trim_brian_monitor(rtm_out_mon, rtm_out_mon.rate, Hz,
                                                          test_range[0], test_range[1])
                rtm_out_t = rtm_out_t * second
                rtm_out_r = gaussian_filter1d(rtm_out_r, self.filter_width / sim_dt) * Hz

                rtm_out_check = rtm_out_r[(rtm_out_t >= detect_min) & (rtm_out_t <= detect_max)]
                if (rtm_out_check > self.min_height).any():
                    check_replay = False
                    check_explosion = True
                    xprint('Replay FAILED: background group exceeded threshold activity', log)
                else:
                    xprint('Background group remained below threshold activity', log)

        xprint('Pulse activities:', log)
        xprint('%s' % (replay_spikes[:]), log)

        xprint('Pulse FWHM [ms]:', log)
        xprint('%s' % (replay_width[:]), log)

        """ SAVE RESULTS in GROUP_REPLAY FILE """

        param_vals = '%d \t ' % options['sim_idx']
        for param_name in group_params:
            param_vals += sim_params[param_name].get_str() + ' \t '

        param_vals += str(stim_events[stim_idx]) + ' \t '

        replay_asy_speed = np.nan
        replay_asy_act = np.nan
        replay_asy_width = np.nan
        if check_replay:
            xprint('Replay check SUCCEEDED!', log)
            if n_asb >= 3:
                replay_asy_speed = (checked_peak_times[-1] - checked_peak_times[-4]) / 3
                replay_asy_act = np.mean(replay_spikes[-3:]) / asb_size
                replay_asy_width = np.mean(replay_width[-3:])

            xprint('with asymptotic speed %.1f ms / asb' % (replay_asy_speed * 1000), log)
            xprint('with asymptotic activity of %.2f' % (replay_asy_act), log)
            xprint('with asymptotic FWHM %.1f ms' % (replay_asy_width), log)

        else:
            xprint('Replay FAILED!', log)

        replay_str = (param_vals + '%s \t %.1f \t %.2f \t %.1f \t %s' %
                      (check_replay, replay_asy_speed * 1000, replay_asy_act,
                       replay_asy_width, check_explosion))

        replay_file = open(options['output_dir'] + options['group_label'] + '/0_group_replay.txt', 'a')
        replay_file.write(replay_str + '\n')
        replay_file.close()


def fit_v_snapshot(v_snapshot, fig_name, log=None, annotate=True):
    """
    Plot voltage snapshot histogram with Gaussian fit.
    
    Args:
        v_snapshot: Array of voltage values in mV
        fig_name: Base filename (without extension) for saving the plot
        log: Optional log file for printing statistics
    
    Returns:
        Dictionary with fit parameters (gauss_mean, gauss_std, gauss_u, gauss_x0, gauss_r2)
    """
    from scipy.optimize import curve_fit
    from general_code.aux_functions import xprint
    
    def gaussian(x, amp, cen, wid):
        return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))
    
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
        
        xprint('\n=========== V GAUSSIAN FIT ===========', log)
        xprint('v (mean +- std) = (%.3f +- %.3f) mV' % (v_mean, v_std), log)
        xprint('gauss mean = %.3f mV' % gauss_mean, log)
        xprint('gauss std = %.3f mV' % gauss_std, log)
        xprint('U = %.3f mV' % gauss_u, log)
        xprint('x0 = %.3f mV' % gauss_x0, log)
        xprint('U - x0 = %.3f mV' % (gauss_u - gauss_x0), log)
        xprint('gauss r2 = %f\n' % gauss_r2, log)
        
    except (RuntimeError, ValueError) as e:
        xprint("WARNING: Couldn't fit voltage distribution to gaussian: %s" % e, log)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(2.5, 2))
    font_size = 8.3
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge', color='black')
    
    v_range = max(v_snapshot) - min(v_snapshot)
    x_values = np.linspace(min(bin_centers) - v_range * 0.05, max(bin_centers) + v_range * 0.05, 100)
    max_plot = -49.8
    
    if fit_gauss and gauss_r2 > 0.5:
        if annotate:
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
    ax.set_ylim([0, 150])
    ax.axvline(-50, color='black', lw=1.5, ls='--')
    ax.set_xticks([-60, -55, -50], labels=['-60', '', '-50'], fontsize=font_size)
    ax.set_xlabel(r'$v$ (mV)', fontsize=font_size)
    ax.set_ylabel('count', fontsize=font_size)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, colors='black', labelsize=font_size)
    
    # Save plot
    fig.savefig(fig_name + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(fig_name + '.svg', bbox_inches='tight')
    plt.close(fig)
    
    return {
        'gauss_mean': gauss_mean,
        'gauss_std': gauss_std,
        'gauss_u': gauss_u,
        'gauss_x0': gauss_x0,
        'gauss_r2': gauss_r2
    }


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
        self.monitor_types = ['v_%d' % asb_idx for asb_idx in asb]

    def perform_test(self, network, settings, test_range, test_data, log=None):

        sim_params = settings.sim_params
        options = settings.options
        group_params = options['group_param_overrides'].keys()

        for pop in self.pops:
            for asb_idx in self.asb:
                stm_name = 'stm_%s_asb_%d_v' % (pop.name, asb_idx)
                
                for time_i in self.time:

                    # create replay properties file
                    v_file_path = options['output_dir'] + options['group_label'] + \
                                  '/0_group_v_pop_%s_%d_%.3fs.txt' % (pop.name, asb_idx, time_i)
                    
                    if not os.path.exists(v_file_path):
                        with open(v_file_path, 'w') as v_file:

                            v_header = 'sim# \t '
                            for param_name in group_params:
                                v_header += param_name + ' \t '

                            v_header += ('v_mean \t v_std \t gauss_mean \t gauss_std \t U \t x0')
                            v_file.write(v_header + '\n')
                            v_file.close()

                    param_vals = '%d \t ' % options['sim_idx']
                    for param_name in group_params:
                        param_vals += sim_params[param_name].get_str() + ' \t '

                    time_arg = np.argmin(np.abs(network[stm_name].t - time_i))
                    v_snapshot = np.array(network[stm_name].v[:, time_arg] / mV)
                    counts, bin_edges = np.histogram(v_snapshot, bins=np.linspace(-60, -50, 41), density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    gauss_mean = np.nan
                    gauss_std = np.nan
                    gauss_u = np.nan
                    gauss_x0 = np.nan

                    v_mean = np.mean(v_snapshot)
                    v_std = np.std(v_snapshot)

                    # Save v_snapshot to file
                    fig_name = (options['output_dir'] + options['group_label'] + '/sim' +
                                str(options['sim_idx']) + '_v%s_%.3fs' % (asb_idx, time_i))
                    np.savetxt(fig_name + '_value.txt', v_snapshot)
                    
                    # Plot using shared function
                    fit_params = fit_v_snapshot(v_snapshot, fig_name, log)
                    gauss_mean = fit_params['gauss_mean']
                    gauss_std = fit_params['gauss_std']
                    gauss_u = fit_params['gauss_u']
                    gauss_x0 = fit_params['gauss_x0']

                    # print to file
                    v_mean = np.mean(v_snapshot)
                    v_std = np.std(v_snapshot)
                    v_str = param_vals + '%.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f' % (
                        v_mean, v_std, gauss_mean, gauss_std, gauss_u, gauss_x0)
                    v_file = open(v_file_path, 'a')
                    v_file.write(v_str + '\n')
                    v_file.close()
