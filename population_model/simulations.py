"""
Discrete-time population-model simulation utilities.

Provides a simple step drive, a discrete simulation runner that advances
assembly activity over time, and helper functions to test and visualize
replay across assemblies.
"""

import os
import numpy as np
from matplotlib import pyplot as plt, rc
from general_code.aux_functions import xprint


rc('mathtext', fontset='stix')
rc('font', family='sans-serif')


class StepDrive:
    """
    Piecewise-constant external drive applied to assembly 1.

    Active with constant amplitude between `start` and `stop` (inclusive).
    """
    def __init__(self, start, stop, amp):
        """
        Initialize the step drive.

        Args:
            start (int): First time step of the drive (inclusive).
            stop (int): Last time step of the drive (inclusive).
            amp (float): Constant amplitude during the active window.
        """
        self.start = start
        self.stop = stop
        self.amp = amp

    def get_drive(self, t_array):
        """
        Return drive array aligned to `t_array` time steps.

        Args:
            t_array (ndarray): Array of integer time steps.

        Returns:
            ndarray: Drive amplitude per time step.
        """
        drive_array = np.zeros(len(t_array))
        drive_array[(t_array >= self.start) & (t_array <= self.stop)] = self.amp
        return drive_array


class DiscreteSimulation:
    """
    Run a discrete-time replay simulation for a given population model.

    Tracks a position field (`vf_array`) and resulting activity
    (`act_array`) for each assembly over time.
    """
    def __init__(self, t_stop, model, ext_drive_asb1=None):
        """

        Args:
            t_stop (int): Final time step (inclusive).
            model: Population model with `vf_next()` and `area_thres()`.
            ext_drive_asb1 (StepDrive|None): Optional external drive for assembly 1.
        """
        self.t_stop = t_stop
        self.t_array = np.arange(0, t_stop + 1)
        self.model = model
        self.n_asb = model.n_asb

        self.vf_array = np.zeros((model.n_asb, t_stop + 1))
        self.act_array = np.zeros((model.n_asb, t_stop + 1))

        if ext_drive_asb1 is not None:
            self.ext_drive_asb1_array = ext_drive_asb1.get_drive(self.t_array)
        else:
            self.ext_drive_asb1_array = np.zeros(t_stop + 1)

    def run_sim(self):
        """
        Advance the simulation one step at a time.

        Updates each assembly's position and activity based on the model
        dynamics, previous activity, and optional external input.
        """
        t_step = 0
        while t_step <= self.t_stop:

            for asb_idx in range(self.model.n_asb):
                # Previous activity (delayed) and external drive
                if asb_idx == 0:
                    act_prev_del = 0
                    ext_drive = self.ext_drive_asb1_array
                else:
                    if t_step == 0:
                        act_prev_del = 0
                    else:
                        act_prev_del = self.act_array[asb_idx - 1, t_step - 1]
                    ext_drive = np.zeros(self.t_stop + 1)

                # Position update and thresholded activity
                self.vf_array[asb_idx, t_step] = self.model.vf_next(self.vf_array[asb_idx, t_step - 1],
                                                                    self.act_array[asb_idx, t_step - 1],
                                                                    act_prev_del,
                                                                    ext_drive[t_step])

                self.act_array[asb_idx, t_step] = self.model.area_thres(self.vf_array[asb_idx, t_step - 1],
                                                                         self.vf_array[asb_idx, t_step])

            t_step += 1


def test_replay(sim, options, sim_params, log=None):
    """
    Compute replay metrics and append to the group results file.

    Args:
        sim (DiscreteSimulation): Simulation object.
        options (dict): Contains output paths and group parameters.
        sim_params (dict): Parameter objects used in the simulation.
        log (str|None): Optional log file path for xprint.
    """

    if hasattr(sim.model, 'n_std'):
        v_success = sim.model.n_std
    else:
        v_success = 1.0

    # create replay properties file
    results_file_path = options['output_dir'] + options['group_label'] + '/0_group_replay.txt'
    if not os.path.exists(results_file_path):
        with open(results_file_path, 'w') as results_file:
            results_header = 'sim# \t '
            group_params = options['group_param_overrides'].keys()
            for param_name in group_params:
                results_header += param_name + ' \t '

            results_header += 'replay \t asb \t std \t cross-time'
            results_file.write(results_header + '\n')
            results_file.close()

    t_cross = np.ones(sim.model.n_asb) * -np.nan
    t_center = np.ones(sim.model.n_asb) * -np.nan
    t_std = np.ones(sim.model.n_asb) * -np.nan
    total_a = np.zeros(sim.model.n_asb)
    for asb_idx in range(sim.model.n_asb):

        # total activity generated
        total_a[asb_idx] = np.sum(sim.act_array[asb_idx])

        # first time distribution crosses success threshold
        post_thr_idx = np.argwhere(sim.vf_array[asb_idx] - v_success >= 0)
        if len(post_thr_idx) > 0:
            t_cross[asb_idx] = sim.t_array[post_thr_idx[0]]

        # activity mean and std (time-domain moments)
        if np.sum(sim.act_array[asb_idx]) > 0:
            t_center[asb_idx] = np.sum(sim.t_array * sim.act_array[asb_idx]) / np.sum(sim.act_array[asb_idx])
            t_variance = np.sum(sim.act_array[asb_idx] * (sim.t_array - t_center[asb_idx])**2) / np.sum(sim.act_array[asb_idx])
            t_std[asb_idx] = np.sqrt(t_variance)
                
    xprint('total activity: %s' % total_a[:], log)
    xprint('v last crossed at: %s' % t_cross[:], log)
    xprint('activity mean: %s' % t_center[:], log)
    xprint('activity std: %s' % t_std[:], log)

    """ SAVE RESULTS in GROUP_RESULTS FILE """
    results_file = open(options['output_dir'] + options['group_label'] + '/0_group_replay.txt', 'r')
    array_param_names = results_file.readline()
    results_file.close()

    param_vals = '%d \t ' % options['sim_idx']
    for param_name in array_param_names.split()[1:]:
        if param_name in sim_params:
            param_vals += sim_params[param_name].get_str() + ' \t '
        elif ((param_name == 'replay') or (param_name == 'asb') or 
              (param_name == 'std') or (param_name == 'cross-time')):
            pass
        else:
            xprint('ERROR: param %s not recognized' % param_name, log)
            param_vals += 'N/A \t '

    results_file = open(options['output_dir'] + options['group_label'] + '/0_group_replay.txt', 'a')

    replay_check = False
    if t_cross[-1] > 0:
        replay_check = True
        xprint('Replay SUCCEEDED!', log)
    else:
        xprint('Replay FAILED!', log)
   
    for asb_idx in range(sim.model.n_asb):
        replay_str = (param_vals + '%s \t %d \t %.3f \t %.3f' %
                        (replay_check, asb_idx + 1, t_std[asb_idx], t_cross[-1]))
        results_file.write(replay_str + '\n')
    results_file.close()


def plot_bars(sim, options, barcolor='#2d8b56cc'):
    """
    Bar plot of assembly activity over time for a single simulation.

    Shows per-assembly activity and cumulative activity, marking when each
    assembly reaches total activity ≥ 1.

    Args:
        sim (DiscreteSimulation): Simulation object with `act_array` and `t_array`.
        options (dict): Contains output directory, labels, and sim index.
        barcolor (str): RGBA color for activity bars.
    """

    fig_width = 0.9 
    fig_height = 1.7
    font_size = 8.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bottom_h = 0
    for asb_idx in range(sim.model.n_asb):
        act_cumsum = np.cumsum(sim.act_array[asb_idx])
        stop_idx = None
        nonzero_idx = np.argwhere(act_cumsum >= 1.)  # mark when cumulative >= 1
        if len(nonzero_idx) > 0:
            stop_idx = nonzero_idx[0]

        act_array = sim.act_array[asb_idx]

        ax.bar(sim.t_array, act_cumsum, color='gray', alpha=0.3, align='center', bottom=bottom_h)

        ax.bar(sim.t_array, act_array, bottom=bottom_h, color=barcolor, align='center')

        if stop_idx is not None:
            ax.bar(sim.t_array[stop_idx], 1.0, bottom=bottom_h, color='none', edgecolor='black', lw=0.5)
            ax.plot([sim.t_array[stop_idx] - 0.35, sim.t_array[stop_idx] + 0.35], [bottom_h + 0.1, bottom_h + 0.9], color='black', lw=0.5)
        bottom_h += 1.1

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

    text_h = 0
    for asb_idx in range(1, 11, 1):
        ax.text(-7, text_h + 0.5, r'$i$ = %d' % asb_idx,
                 fontsize=font_size, ha='left', va='center', c='k')
        text_h += 1.1

    fig_dir = (options['output_dir'] + options['group_label'] + 
               '/sim%d' % (options['sim_idx']) +
               '_' + options['group_param_array_str'])
    plt.savefig(fig_dir + '_replay_bars.png', dpi=300, bbox_inches='tight')
