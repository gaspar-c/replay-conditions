import brian2
from brian2 import *
import os
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import scipy.optimize
import numpy as np
import time
import json
import hashlib


def gen_binary_array(array_size, prob):
    """
    generate boolean array where each element has probability 'prob' of being True
    'prob' precision must not be smaller than 1/integer_scale
    """

    integer_scale = 1000

    if int(prob*integer_scale) == 0:
        print('Error, probability precision is too small!')
        exit()

    rand_array = np.random.randint(integer_scale, size=array_size, dtype=np.uint16)
    return rand_array < integer_scale*prob


def xprint(string, logfile=None):
    print(string)

    if logfile is not None:
        logger = open(logfile, 'a')
        logger.write(string + '\n')
        logger.close()


class ConnectivityMatrix:
    def __init__(self, p_ij, n_j, n_i, name, fixed_size=False, to_file=False):
        """
        create connectivity matrix object
        Args:
            p_ij: probability of J->I connection
            n_j: number of J cells
            n_i: number of I cells
            name: name of connection
            fixed_size: [True/False], determines if the number of synapses is
                        always fixed (equal to the expected value)
            to_file: [True/False], determines if connectivity object is stored to a file
        """

        self.p_ij = p_ij
        self.n_j = int(n_j)
        self.n_i = int(n_i)
        self.name = name

        self.seed = None
        self.pre_idx = None   # indices of presynaptic neurons
        self.post_idx = None  # indices of postsynaptic neurons
        self.n_syn = 0
        self.fixed_size = fixed_size
        self.to_file = to_file

    def create_conn(self, seed, log=None):
        self.seed = seed

        np.random.seed(self.seed)

        # if fixed in-degree synapses:
        if self.fixed_size:

            n_syn_per_post = int(self.n_j * self.p_ij)
            self.n_syn = n_syn_per_post * self.n_i

            total_start_time = time.time()
            self.pre_idx = np.random.choice(np.arange(self.n_j), size=self.n_syn, replace=True)
            self.post_idx = np.repeat(np.arange(self.n_i), n_syn_per_post)
            xprint('\t Calculated %s synapses in %.0f seconds with fixed in-degree. '
                   'Each post-synaptic neuron received %s connections' %
                   ('{:,}'.format(self.n_syn), (time.time() - total_start_time), '{:,}'.format(n_syn_per_post)),
                   log)

        # if not fixed size:
        else:
            # connect J->I
            total_size = self.n_j * self.n_i

            # initialise array to store synapses with 1.5x the expected size
            # if int(total_size) > 4294967295:
            #     print('too many synapses: type uint32 is not big enough!')
            #     exit()
            synapses = np.zeros(int(total_size * self.p_ij * 1.5), dtype=np.uint64)

            # make max_size smaller if computer runs out of space for calculation
            # the smaller the number, the longer the calculation will take
            max_size = int(2 * 1e9)

            total_start_time = time.time()
            count_step = 0
            count_syn = 0
            size_left = total_size
            if total_size > max_size:
                time_spent = 0

                xprint('\t calculating ca. %s out of %s (%s x %s) possible synapses...' %
                       ('{:,}'.format(int(total_size * self.p_ij)), '{:,}'.format(total_size),
                        '{:,}'.format(self.n_j), '{:,}'.format(self.n_i)),
                        log)
                while size_left > max_size:
                    start_time0 = time.time()

                    conn_matrix_part = gen_binary_array(max_size, self.p_ij)
                    non_zeros_part = count_step * max_size + np.nonzero(conn_matrix_part)[0]
                    synapses[count_syn:count_syn + len(non_zeros_part)] = non_zeros_part
                    count_syn += len(non_zeros_part)
                    count_step += 1

                    # estimate time:
                    size_left = size_left - max_size
                    curr_percent = (1 - size_left / total_size) * 100
                    time_spent += (time.time() - start_time0)
                    time_left = (100 - curr_percent) * time_spent / curr_percent
                    xprint('\t\t %s: %.0f%% calculated in %.0f seconds (ca. %.0f seconds left...)' %
                           ('{:,}'.format(count_syn), curr_percent, time_spent, time_left), log)

            conn_matrix_part = gen_binary_array(size_left, self.p_ij)
            non_zeros_part = count_step * max_size + np.nonzero(conn_matrix_part)[0]

            synapses[count_syn:count_syn + len(non_zeros_part)] = non_zeros_part
            count_syn += len(non_zeros_part)

            synapses = synapses[:count_syn]

            xprint('\t %s: calculated all synapses in %.0f seconds.' %
                   ('{:,}'.format(count_syn), (time.time() - total_start_time)), log)

            self.pre_idx = np.array(synapses % self.n_j, dtype=int)
            self.post_idx = np.array(synapses // self.n_j, dtype=int)
            self.n_syn = len(self.pre_idx)


def flat_window(x, width):
    return np.piecewise(x, [(x >= -width/2) & (x < width/2)], [1/width, 0])


def filter_array(array, dt, filter_func, filter_width):
    """
    filters an array with a given filter function
    Args:
        array: array to be filtered
        dt: time step
        filter_func: filter kernel
        filter_width: width of filter kernel

    Returns:
        filtered array

    """
    if filter_width / second > 0:
        if filter_func == 'gauss':
            array = gaussian_filter1d(array, filter_width/dt)

        elif filter_func == 'flat':
            t = np.arange(-filter_width/dt, filter_width/dt + 1, 1)
            array = np.convolve(array, flat_window(t, filter_width/dt), mode='same')
        else:
            print('ERROR: smooth type not recognized!')

    return array


def calc_rate_monitor(spike_monitor, t_start, t_stop, dt):
    """
    calculate rate monitor from brian2 spiking monitor

    Args:
        spike_monitor: brian2 spiking monitor object
        t_start: calculation start time
        t_stop: calculation stop time
        dt: simulation step time

    Returns:
        time_array: rate monitor time array
        rate_monitor: rate monitor array

    """
    n_cells = len(spike_monitor.spike_trains())

    spike_times = spike_monitor.t[(spike_monitor.t >= t_start) & (spike_monitor.t <= t_stop)]

    time_array = np.arange(t_start / second, (t_stop + dt)/ second, dt/second)

    spike_counts = np.zeros(len(time_array))
    unique_times, unique_counts = np.unique(spike_times, return_counts=True)
    unique_idx = np.array((unique_times - t_start)/dt, dtype=int)
    spike_counts[unique_idx] = unique_counts

    rtm_array = spike_counts/((dt/second) * n_cells)

    return time_array, rtm_array


def lorentzian(x_array, a, mu, sigma):
    """
    the lorentzian function.

    Args:
        x_array: argument of function
        a: parameter 1
        mu: parameter 2, peak center
        sigma: parameter 3, width at half maximum

    Returns:
        output: the lorentzian function
    """

    output = (a / np.pi) * sigma / ((x_array - mu) ** 2 + sigma ** 2)
    return output


def get_q_factor(a, sigma):
    """
    calculate Q-factor of a lorentzian

    Args:
        a: parameter 1 of Lorentzian function
        sigma: parameter 3 of Lorentzian function

    Returns:
        q_factor
    """

    peak = a / (np.pi * sigma)
    fwhm = 2 * sigma
    q_factor = peak / fwhm

    return q_factor


def calc_network_frequency(pop_rate, sim_time, dt, max_freq, fit=True):
    """
    calculate Power Spectral Density (PSD) of population activity
    and try to fit it to a lorentzian function

    Args:
        pop_rate: population rate signal
        sim_time: total time of pop_rate signal
        dt: time step of pop_rate signal
        max_freq: maximum frequency of the power spectrum
        fit: [true/false] if true, try to fit PSD to lorentzian

    Returns:
        fft_freq: frequency array of FFT
        fft_psd: PSD array of FFT
        fit_params: estimated parameters of lorentzian fit
    """

    # Power Spectral Density (PSD) (absolute value of Fast Fourier Transform) centered around the mean:
    fft_psd = np.abs(np.fft.fft(pop_rate - np.mean(pop_rate)) * dt) ** 2 / (sim_time / second)

    # frequency arrays for PSD:
    fft_freq = np.fft.fftfreq(pop_rate.size, dt)

    # delete second (mirrored) half of PSD:
    fft_psd = fft_psd[:int(fft_psd.size / 2)]
    fft_freq = fft_freq[:int(fft_freq.size / 2)]

    # find argument where frequency is closest to max_freq:
    arg_lim = (np.abs(fft_freq - max_freq)).argmin()

    # select power spectrum range [1,max_freq] Hz:
    fft_freq = fft_freq[1:arg_lim + 1]
    fft_psd = fft_psd[1:arg_lim + 1]

    fit_params = []
    if fit and (fft_psd != 0).any():

        # find maximum or power spectrum:
        i_arg_max = np.argmax(fft_psd)
        freq_max = fft_freq[i_arg_max]
        psd_max = fft_psd[i_arg_max]

        # fit power spectrum peak to Lorentzian function:
        try:
            fit_params, _ = scipy.optimize.curve_fit(lorentzian, fft_freq, fft_psd,
                                                     p0=(psd_max, freq_max, 1), maxfev=1000)
        except RuntimeError:
            pass
        finally:
            if fit_params is []:
                print("WARNING: Couldn't fit PSD to Lorentzian")

    return fft_freq, fft_psd, fit_params


def check_brian_monitor(network, mon_name, mon_attr):
    """
    check if a given brian monitor with
    a given attribute (variable being measured) exists in the network object

    Args:
        network: brian network object
        mon_name: name of monitor to check
        mon_attr: name of monitor attribute to check

    Returns:
        check: [true/false]
    """

    check = False
    if mon_name in network:
        monitor = network[mon_name]
        if hasattr(monitor, mon_attr):
            check = True

    return check


def trim_brian_monitor(monitor, attr, attr_unit, t_start, t_stop):
    """
    trim a given brian monitor attribute to a given time range

    Args:
        monitor: brian monitor object
        attr: attribute of monitor (variable being measured)
        attr_unit: output unit of attribute
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        time_array: trimmed time array
        attr_array: trimmed attribute array (unit-less)
    """

    time_array = np.array(monitor.t / second)[(monitor.t >= t_start) &
                                              (monitor.t <= t_stop)]

    attr_array = np.array(attr / attr_unit)[(monitor.t >= t_start) &
                                            (monitor.t <= t_stop)]

    return time_array, attr_array


def trim_brian_monitor_group(monitor, attr, attr_unit, t_start, t_stop):
    """
    trim a given brian monitor attribute to a given time range

    Args:
        monitor: brian monitor object
        attr: attribute of monitor (variable being measured)
        attr_unit: output unit of attribute
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        time_array: trimmed time array
        attr_array: trimmed attribute array (unit-less)
    """

    time_array = np.array(monitor.t / second)[(monitor.t >= t_start) &
                                              (monitor.t <= t_stop)]

    attr_array = np.array(attr[:, (monitor.t >= t_start) &
                                  (monitor.t <= t_stop)] / attr_unit)

    return time_array, attr_array


def calc_low_pass_filter(data, cutoff, dt):
    """
    calculate low pass filter

    Args:
        data: input signal
        cutoff: cutoff frequency of low pass filter
        dt: time step of data signal

    Returns:
        low_pass_trace: low-pass-filtered signal
    """

    nyquist_freq = (1/2) * (1/dt)
    b_butter, a_butter = signal.butter(2, cutoff/nyquist_freq, btype='low', analog=False, output='ba')

    low_pass_trace = signal.filtfilt(b_butter, a_butter, data)

    return low_pass_trace


def param_array_str(param_array):
    """
    creates string with parameters in a given array and their values

    Args:
        param_array: array with parameters

    Returns:
        string with parameter arrays and their values

    """
    out_str = ''
    k = 0
    for param in param_array:
        if type(param_array[param]) is tuple:
            param_val = str(param_array[param][0]) + str(param_array[param][1])
        else:
            param_val = param_array[param]

        if k > 0:
            out_str += '_'
        out_str += param + '_%s' % param_val
        k += 1

    return out_str


def clear_brian_caches(clear_cython=False, clear_tex=False):
    """
    clears brian caches
    """
    if clear_cython:
        cython_path = '~/.cython/brian_extensions'
        if os.path.exists(cython_path):
            brian2.clear_cache('cython')

    if clear_tex:
        tex_path = '~/.cache/matplotlib/tex.cache'
        if os.path.exists(tex_path):
            os.remove(tex_path)

def seconds_to_hhmmss(seconds):
    """
    converts input in seconds to str in format 'hh:mm:ss'
    """
    string_out = ''
    hh = seconds // 3600
    if hh > 0:
        string_out += ('%d' % hh) + 'h '

    remainder = seconds % 3600
    mm = remainder // 60
    if mm > 0:
        string_out += ('%d' % mm) + 'm '

    ss = int(remainder % 60)
    string_out += ('%d' % ss) + 's '

    return string_out


def gaussian_function(x_array, a, b, c):
    return a * np.exp(- (x_array - b) ** 2 / (2 * (c ** 2)))


def get_unit_rates(test_data, spm_mon, test_range, pop_name, log=None):
    """
    calculate individual firing rate for each neuron

    Args:
        test_data: test_results
        spm_mon: brian spike monitor object
        test_range: calculation time range
        pop_name: name of neuron population
        log: log file
    """

    t_start = test_range[0]
    t_stop = test_range[1]
    spike_trains = spm_mon.spike_trains()
    num_neurons = len(spike_trains)
    num_spikes = np.zeros(num_neurons)

    for i in range(num_neurons):
        num_spikes[i] = len((spike_trains[i])[(spike_trains[i] >= t_start) &
                                              (spike_trains[i] <= t_stop)])

    all_rates = num_spikes / (t_stop - t_start)

    mean_rate = np.mean(all_rates) / Hz
    std_rate = np.std(all_rates) / Hz

    test_data['mean_rate_' + pop_name] = mean_rate
    test_data['std_rate_' + pop_name] = std_rate
    test_data['all_rates_' + pop_name] = all_rates
    xprint('%s firing rate (%.2f +/- %.2f) Hz' % (pop_name.upper(), mean_rate, std_rate), log)


def get_isi_cv(test_data, spm_mon, test_range, pop_name, log=None):
    """
    calculate the Coefficient of Variation (CV) of
    the Inter-Spike-Interval (ISI) for each neuron

    Args:
        test_data: test results
        spm_mon: brian spike monitor
        test_range: calculation time range
        pop_name: name of neuron population
        log: log file
    """

    mean_isi_cv = 0
    std_isi_cv = 0
    t_start = test_range[0]
    t_stop = test_range[1]

    # get spike times for whole network:
    spike_trains = spm_mon.spike_trains()

    # trim spikes within calculation time range:
    num_neurons = len(spike_trains)
    cut_spike_trains = {new_list: [] for new_list in range(num_neurons)}
    for i in range(num_neurons):
        cut_spike_trains[i] = (spike_trains[i])[(spike_trains[i] >= t_start) &
                                                (spike_trains[i] <= t_stop)]

    # check if at least one neuron spikes at least twice in the selected interval:
    n = 0
    for i in range(num_neurons):
        if (cut_spike_trains[i]).size >= 2:
            n += 1

    # n is the number of neurons that spiked at least twice.
    # if there is at least one of those:
    if n > 0:
        check_enough_spikes = True

        # calculate ISI CV for each neuron:
        all_isi_cvs = np.zeros(n)
        j = 0
        for i in range(num_neurons):
            # if the neuron spiked at least twice:
            if (cut_spike_trains[i]).size >= 2:
                # get array of ISIs:
                isi = np.diff(cut_spike_trains[i])
                # calculate average and std of ISIs:
                avg_isi = np.mean(isi)
                std_isi = np.std(isi)
                # store value of neuron ISI CV
                all_isi_cvs[j] = std_isi / avg_isi
                j = j + 1

        # calculate mean and std of all ISI CVs:
        mean_isi_cv = np.mean(all_isi_cvs)
        std_isi_cv = np.std(all_isi_cvs)

    # if not enough spikes to perform calculation:
    else:
        check_enough_spikes = False

    if check_enough_spikes:
        test_data['mean_cv_' + pop_name] = mean_isi_cv
        test_data['std_cv_' + pop_name] = std_isi_cv
        xprint('%s ISI CV = %.2f +/- %.2f' % (pop_name.upper(), mean_isi_cv, std_isi_cv), log)
    else:
        xprint('%s ISI CV could not be calculated!' % pop_name.upper(), log)


def get_synchronicity(test_data, spm_mon, test_range, pop_name, max_freq, log=None):
    """
    calculates neuron population rate and its Power Spectral Density (PSD), which is
    fitted to a lorentzian function. The quality of the fit determines whether the
    population has an underlying oscillation

    Args:
        test_data: test results
        spm_mon: brian spike monitor
        test_range: calculation time range
        pop_name: name of neuron population
        max_freq: maximum frequency for PSD
        log: log file
    """

    sim_dt = defaultclock.dt

    fit_worked = False
    _, raw_rtm = calc_rate_monitor(spm_mon, test_range[0], test_range[1], sim_dt)

    # get power spectrum of population rate in the [0, max_freq] range and fit it to a lorentzian:
    _, _, lorentz_fit = calc_network_frequency(raw_rtm, test_range[1] - test_range[0],
                                               sim_dt / second, max_freq / Hz, fit=True)

    # calculate q-factor of lorentzian fit:
    if len(lorentz_fit) > 0:
        lorentz_a, lorentz_mu, lorentz_sigma = lorentz_fit
        if lorentz_mu >= 0:
            q_factor = get_q_factor(lorentz_a, lorentz_sigma)
            if q_factor >= 0.01:
                fit_worked = True
                test_data['q_factor_' + pop_name] = q_factor
                test_data['net_freq_' + pop_name] = lorentz_mu

                xprint('%s network freq. %.2f Hz; Q-factor = %.2f' %
                       (pop_name.upper(), lorentz_mu, q_factor), log)

    if not fit_worked:
        xprint('%s is Asynchronous: Q-factor could not be calculated' % pop_name.upper(), log)


def square_func(x, mu, width):
    out = np.zeros(len(x))
    out[(x > mu - width/2) & (x <= mu + width/2)] = 1 / width
    return out
