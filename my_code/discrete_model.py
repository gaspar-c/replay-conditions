from brian2 import *
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import scipy.special
import scipy.stats
from my_code.aux_functions import xprint
from my_code.plots import color_gradient
from my_code.parameters import Parameter
import warnings


def simplified_theory(ff, rc, asb_idx, t_step, init_j=1.0, q_inf=False, speed=None, v_drag=0.):

    # finite_s
    if q_inf:
        ff_2d = ff[:, None]
        rc_2d = rc[None, :]
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # Ignore division by zero
            v_inf = 1 / (1 - rc_2d)
            # if np.isfinite(speed):
            #     v_small = v_inf
            # else:
            v_small = (1 / speed) * ((rc_2d / (1 - speed)) ** (1 / speed - 1))

        v = ff_2d * np.where(rc_2d < 1 - speed, v_inf, v_small)
        out = np.zeros_like(v, dtype=bool)
        out[v >= 1] = True
    else:
        s_step = t_step - asb_idx
        if np.isfinite(t_step):
            warnings.filterwarnings("error")
            k_array = np.arange(0, s_step + 1)
            binom_coeff_log = np.log(scipy.special.comb(k_array + asb_idx - 1, asb_idx - 1))
            rc_2d = rc[:, np.newaxis]
            power_terms_log = k_array * np.log(rc_2d)
            ff_coeff_log = (asb_idx - 1) * np.log(ff)

            if np.array(ff).size == 1:
                ff_coeff_log_term = ff_coeff_log
            else:
                ff_coeff_log_term = ff_coeff_log[:, None, None]

            sum_terms_log = ff_coeff_log_term + binom_coeff_log[None, None, :] + power_terms_log[None, :, :]

            with np.errstate(over='ignore'):
                sum_terms = np.exp(sum_terms_log)
                v = np.sum(sum_terms, axis=2)

            v[np.argwhere(np.isnan(v))] = np.finfo(np.float64).max

            # print(sum_terms)
            # print(sum_terms[np.isfinite(sum_terms)])
            # print(np.argwhere(np.isfinite(v)))
            # exit()

        # infinite s
        else:
            ff_2d = ff[:, None]
            rc_2d = rc[None, :]
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):  # Ignore division by zero
                v_small = ((ff_2d / (1 - rc_2d)) ** asb_idx) / ff_2d

            v = np.where(rc_2d >= 1, np.finfo(np.float64).max, v_small)

        out = np.zeros_like(v, dtype=bool)
        out[v >= 1/init_j] = True

    return out


class StepDrive:
    def __init__(self, start, stop, amp):
        self.start = start
        self.stop = stop
        self.amp = amp

    def get_drive(self, t_array):
        drive_array = np.zeros(len(t_array)) * mV
        drive_array[(t_array >= self.start) & (t_array <= self.stop)] = self.amp
        return drive_array


class RectangleModel:
    def __init__(self, n_asb,
                 w_rc=0.*mV, w_ff=0.*mV,
                 vf_init=0.*mV, v_thres=0.*mV, v_width=0.*mV,
                 v_decay=0*mV,
                 inf_model=False):
        self.n_asb = n_asb
        self.w_rc = w_rc
        self.w_ff = w_ff
        self.vf_init = vf_init
        self.v_thres = v_thres
        self.v_width = v_width
        self.v_decay = v_decay
        self.inf_model = inf_model

    def vf_next(self, vf, rate_del, rate_prev_del, drive_ext):
        out = vf
        if vf < self.v_thres + self.v_width:
            # decay, if not fully crossed and previous v had already moved beyond the threshold
            if (vf > self.v_thres) and (vf < self.v_thres + self.v_width):
                out -= self.v_decay

            # move to the right
            out += self.w_rc * rate_del + self.w_ff * rate_prev_del + drive_ext
        else:
            if self.inf_model:
                # move to the right
                out += self.w_rc * rate_del + self.w_ff * rate_prev_del + drive_ext


        if not self.inf_model:
            # saturate with some numerical precision (0.0001% of width)
            if out >= self.v_thres + self.v_width:
                out = self.v_thres + self.v_width * 1.000001

        return out

    # get area that crossed the threshold:
    def area_thres(self, vf_prev, vf_curr):
        act = 0
        diff = (vf_curr - vf_prev) / self.v_width
        if diff > 1:
            diff = 1
        if diff > 0:
            act = diff
        return act


class GaussModel:
    def __init__(self, n_asb,
                 w_rc=0.*mV, w_ff=0.*mV,
                 vf_init=0.*mV, v_thres=0.*mV, v_width=0.*mV,
                 n_std=1,
                 inf_model=False):
        self.n_asb = n_asb
        self.w_rc = w_rc
        self.w_ff = w_ff
        self.vf_init = vf_init
        self.v_thres = v_thres
        self.v_width = v_width
        self.n_std = n_std
        self.inf_model = inf_model

    def vf_next(self, v_pos, rate_del, rate_prev_del, drive_ext):
        out = v_pos

        # move to the right
        out += self.w_rc * rate_del + self.w_ff * rate_prev_del + drive_ext

        # saturate with some numerical precision (0.001% of width)
        if out >= self.v_thres + self.v_width * self.n_std:
            out = self.v_thres + self.v_width * self.n_std * 1.000001

        return out

    # get area that crossed the threshold:
    def area_thres(self, v_prev, v_curr):
        act = 0
        total_area = m.erf(self.n_std / np.sqrt(2))
        area_prev = 1/2 * (1 - m.erf((self.v_thres - v_prev) / (np.sqrt(2) * self.v_width))) / total_area
        area_curr = 1/2 * (1 - m.erf((self.v_thres - v_curr) / (np.sqrt(2) * self.v_width))) / total_area
        diff = area_curr - area_prev
        if diff > 1:
            diff = 1
        if diff > 0:
            act = diff
        return act



class DiscreteSimulation:
    def __init__(self, t_stop, model, ext_drive_asb1=None):
        self.t_stop = t_stop
        self.t_array = np.arange(0, t_stop + 1)
        self.model = model
        self.n_asb = model.n_asb

        self.vf_array = np.ones((model.n_asb, t_stop + 1)) * self.model.vf_init
        self.rate_array = np.zeros((model.n_asb, t_stop + 1))

        if ext_drive_asb1 is not None:
            self.ext_drive_asb1_array = ext_drive_asb1.get_drive(self.t_array)
        else:
            self.ext_drive_asb1_array = np.zeros(t_stop + 1) * mV

    def run_sim(self):
        t_step = 0
        while t_step <= self.t_stop:

            for asb_idx in range(self.model.n_asb):
                if asb_idx == 0:
                    rate_prev_del = 0
                    ext_drive = self.ext_drive_asb1_array
                else:
                    if t_step == 0:
                        rate_prev_del = 0
                    else:
                        rate_prev_del = self.rate_array[asb_idx - 1, t_step - 1]
                    ext_drive = np.zeros(self.t_stop + 1) * mV

                self.vf_array[asb_idx, t_step] = self.model.vf_next(self.vf_array[asb_idx, t_step - 1],
                                                                    self.rate_array[asb_idx, t_step - 1],
                                                                    rate_prev_del,
                                                                    ext_drive[t_step])

                self.rate_array[asb_idx, t_step] = self.model.area_thres(self.vf_array[asb_idx, t_step - 1],
                                                                         self.vf_array[asb_idx, t_step])

            t_step += 1


def plot_bars(sim, options, plot_total=True, fold=True, barcolor='#2d8b56cc'):

    fig_width = 0.9  #* (len(sim.t_array) / 15)
    fig_height = 1.7 #* (sim.model.n_asb / 10)
    font_size =8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))


    bottom_h = 0
    # low_lim = 0.005

    for asb_idx in range(sim.model.n_asb):
        act_cumsum = np.cumsum(sim.rate_array[asb_idx])
        # print(act_cumsum)
        # exit()
        stop_idx = None
        nonzero_idx = np.argwhere(act_cumsum >= 1.)
        if len(nonzero_idx) > 0:
            stop_idx = nonzero_idx[0]
            # act_cumsum[stop_idx + 1:] = 0   #  don't plot gray bars after successful

        act_array = sim.rate_array[asb_idx]

        if plot_total:
            ax.bar(sim.t_array, act_cumsum, color='gray', alpha=0.3,
                   align='center', bottom=bottom_h)
            # ax.bar(np.arange(asb_idx + 1, stop_idx + 1), 1.0, color='none', edgecolor='black', lw=0.5,
            #        align='center', bottom=bottom_h)

        ax.bar(sim.t_array, act_array, bottom=bottom_h, color=barcolor, align='center')

        if stop_idx is not None:
            ax.bar(sim.t_array[stop_idx], 1.0, bottom=bottom_h, color='none', edgecolor='black', lw=0.5)
            ax.plot([sim.t_array[stop_idx] - 0.35, sim.t_array[stop_idx] + 0.35], [bottom_h + 0.1, bottom_h + 0.9], color='black', lw=0.5)
            # ax.plot([sim.t_array[stop_idx] - 0.3, sim.t_array[stop_idx] + 0.3], [bottom_h + 1, bottom_h], color='black', lw=1.0)
        bottom_h += 1.1

        if fold and bottom_h > 10:
            bottom_h = 0

    ax.set_xlabel(r'time step', fontsize=font_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=2)

    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    x_ticks = [1, 5, 10, 15]
    ax.set_xticks(x_ticks)
    ax.set_yticks([])
    ax.set_xticklabels(x_ticks, fontsize=font_size, c='k')
    ax.set_yticklabels(labels=[])

    # ax_curr.set_ylabel('Ext')
    # ax.set_ylabel('Activity', fontsize=font_size)

    text_h = 0
    for asb_idx in range(1, 11, 1):
        ax.text(-7, text_h + 0.5, r'$i$ = %d' % asb_idx, fontsize=font_size, ha='left', va='center', c='k')
        text_h += 1.1

    # plt.tight_layout()
    fig_dir = (options['output_dir'] + options['time_stamp'] + '/sim%d' % (options['sim_idx']) +
               '_' + options['group_param_array_str'])
    plt.savefig(fig_dir + '_replay_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir + '_replay_bars.svg', dpi=600, bbox_inches='tight')


def test_replay(sim, options, sim_params, log=None, onlylast=False):

    if hasattr(sim.model, 'n_std'):
        v_success = sim.model.n_std * sim.model.v_width
    else:
        v_success = sim.model.v_width

    # create replay properties file
    results_file_path = options['output_dir'] + options['time_stamp'] + '/0_group_square.txt'
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w') as results_file:
            results_header = 'sim# \t '
            group_params = options['group_param_array'].keys()
            for param_name in group_params:
                results_header += param_name + ' \t '

            results_header += 'replay \t asb \t cross \t mean \t std \t cross-time \t iqr \t total-a'
            results_file.write(results_header + '\n')
            results_file.close()

    t_cross = np.ones(sim.model.n_asb) * -np.nan
    t_center = np.ones(sim.model.n_asb) * -np.nan
    t_std = np.ones(sim.model.n_asb) * -np.nan
    t_iqr = np.ones(sim.model.n_asb) * -np.nan
    total_a = np.zeros(sim.model.n_asb)
    for asb_idx in range(sim.model.n_asb):

        # get total activity generated
        total_a[asb_idx] = np.sum(sim.rate_array[asb_idx])

        # get v crossing point:
        post_thr_idx = np.argwhere(sim.vf_array[asb_idx] - v_success - sim.model.v_thres >= 0)
        if len(post_thr_idx) > 0:
            t_cross[asb_idx] = sim.t_array[post_thr_idx[0]]

        # get activity moments:
        if np.sum(sim.rate_array[asb_idx]) > 0:
            t_center[asb_idx] = np.sum(sim.t_array * sim.rate_array[asb_idx]) / np.sum(sim.rate_array[asb_idx])
            t_variance = np.sum(sim.rate_array[asb_idx] * (sim.t_array - t_center[asb_idx])**2) / np.sum(sim.rate_array[asb_idx])
            t_std[asb_idx] = np.sqrt(t_variance)


            if np.sum(sim.rate_array[asb_idx]) >= 1:
                cdf = np.cumsum(sim.rate_array[asb_idx])

                # Define the quantiles
                quantiles = [0.25, 0.75]

                # Initialize lists to store the results
                quantile_times = []

                # Loop over each quantile
                for q in quantiles:
                    # Find where the CDF crosses the quantile
                    idx = np.searchsorted(cdf, q)

                    if idx == 0:
                        # Quantile is before the first time point
                        quantile_time = sim.t_array[0]
                    else:
                        # Interpolate between the previous and current time points
                        cdf_prev = cdf[idx - 1]
                        cdf_curr = cdf[idx]
                        time_prev = sim.t_array[idx - 1]
                        time_curr = sim.t_array[idx]

                        if cdf_curr == cdf_prev:
                            # All the weight is at this time point
                            quantile_time = time_curr
                        else:
                            # Linear interpolation
                            quantile_time = time_prev + (q - cdf_prev) * (time_curr - time_prev) / (cdf_curr - cdf_prev)

                    quantile_times.append(quantile_time)

                Q1_time, Q3_time = quantile_times
                IQR = Q3_time - Q1_time
                t_iqr[asb_idx] = IQR

    xprint('total activity: %s' % total_a[:], log)
    xprint('v last crossed at: %s' % t_cross[:], log)
    xprint('activity mean: %s' % t_center[:], log)
    xprint('activity std: %s' % t_std[:], log)
    xprint('activity IQR: %s' % t_iqr[:], log)

    """ SAVE RESULTS in GROUP_RESULTS FILE """
    results_file = open(options['output_dir'] + options['time_stamp'] + '/0_group_square.txt', 'r')
    array_param_names = results_file.readline()
    results_file.close()

    param_vals = '%d \t ' % options['sim_idx']
    for param_name in array_param_names.split()[1:]:
        if param_name in sim_params:
            param_vals += sim_params[param_name].get_str() + ' \t '
        elif ((param_name == 'replay') or (param_name == 'asb') or (param_name == 'cross') or
              (param_name == 'mean') or (param_name == 'std') or (param_name == 'cross-time') or
              (param_name == 'iqr') or (param_name == 'total-a')):
            pass
        else:
            xprint('ERROR: param %s not recognized' % param_name, log)
            param_vals += 'N/A \t '

    results_file = open(options['output_dir'] + options['time_stamp'] + '/0_group_square.txt', 'a')

    replay_check = False
    if t_cross[-1] > 0:
        replay_check = True
        xprint('Replay SUCCEEDED!', log)
    else:
        xprint('Replay FAILED!', log)

    if onlylast:
        replay_square_str = (param_vals + '%s \t %d \t %s \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f' %
                            (replay_check, sim.model.n_asb,
                             t_cross[-1], t_center[-1], t_std[-1], t_cross[-1], t_iqr[-1], total_a[-1]))
        results_file.write(replay_square_str + '\n')
    else:
        for asb_idx in range(sim.model.n_asb):
            replay_square_str = (param_vals + '%s \t %d \t %s \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f' %
                            (replay_check, asb_idx + 1,
                             t_cross[asb_idx], t_center[asb_idx], t_std[asb_idx], t_cross[-1],
                             t_iqr[asb_idx], total_a[asb_idx]))
            results_file.write(replay_square_str + '\n')
    results_file.close()
