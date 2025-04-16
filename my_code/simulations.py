from brian2 import *
import csv
import random
import os
import socket
import multiprocessing
from multiprocessing import Pool
from functools import partial
import time
from my_code.parameters import get_dft_sim_params, get_dft_plot_params, load_param_array, print_attr
from my_code.aux_functions import (xprint, clear_brian_caches, seconds_to_hhmmss,
                                   param_array_str)
from my_code.tests import NetworkTests, MonitorSettings
from my_code.network import init_network, VoltageStimulus, ChangeAttribute


def choose_from_group_params(settings, group_params, sim_idx, n_cores):
    """
    determine which parameters will be used for each simulation
    Args:
        settings: simulation settings
        group_params: parameters for group of simulations
        sim_idx: simulation index (w.r.t. group of simulations)
        n_cores: number of cpu cores

    Returns:
        simulation settings
    """
    sim_settings = settings.copy()

    sim_settings['sim_idx'] = sim_idx
    sim_settings['core_idx'] = (sim_idx % n_cores) + 1

    select_params = {}
    for param_name in group_params:
        param_array = group_params[param_name]
        if len(param_array) > (sim_idx - 1):
            select_params[param_name] = param_array[sim_idx - 1]
        else:
            select_params[param_name] = param_array[-1]
    sim_settings['group_param_array'] = select_params
    sim_settings['group_param_array_str'] = param_array_str(select_params)

    return sim_settings


class SimElement:
    """
    object determining simulation properties
    """
    def __init__(self,
                 options,
                 brian_net,
                 net_objects,
                 sim_params,
                 plot_params,
                 events,
                 monitors,
                 log):
        self.options = options
        self.brian_net = brian_net
        self.net_objects = net_objects
        self.sim_params = sim_params
        self.plot_params = plot_params
        self.events = events
        self.monitors = monitors
        self.log = log


def run_sim_group(group_options, group_params, run_single):
    """
    run group of simulations. This function distributes the specified
    settings and parameters for the whole simulation group to each individual
    simulation, running each of them in parallel (if several CPUs are available)
    """

    """ CREATE SIMULATION GROUP OUTPUT FOLDER """
    group_log = group_options['output_dir'] + group_options['time_stamp'] + '/0_group_log.log'
    os.mkdir(group_options['output_dir'] + group_options['time_stamp'])

    """ MAKE N SIMS AS LARGE AS GROUP PARAMETER ARRAY """
    n_sims = 1

    for param_name in group_params:
        if len(group_params[param_name]) > n_sims:
            n_sims = len(group_params[param_name])

    group_options['n_sims'] = n_sims
    host_name = socket.gethostname()
    script_name = os.path.basename(sys.argv[0])
    xprint('Running %d simulations from %s in %s...' % (group_options['n_sims'], script_name, host_name), group_log)

    """ ATTRIBUTE SIM INDEX AND CPU CORE """
    n_sims = group_options['n_sims']
    cpu_cores = multiprocessing.cpu_count()
    max_cores = min([group_options['max_cores'], cpu_cores])
    n_cores = min([n_sims, max_cores])

    """ CREATE TABLE WITH PARAM ARRAY IN GROUP LOG """
    lines = [None] * (n_sims + 1)

    # Header
    lines[0] = ['sim#']
    for param_name in group_params:
        lines[0].append(param_name)

    # Param array
    for i in range(n_sims):
        k = i + 1
        lines[k] = [str(k)]
        for param_name in group_params:
            if i < len(group_params[param_name]):
                val_param = group_params[param_name][i]
            else:
                val_param = group_params[param_name][-1]
            lines[k].append(str(val_param))

    # Print to log
    for k in range(len(lines)):
        xprint('\t\t'.join(lines[k]), group_log)

    """ RUN SIMULATIONS """
    start_time = time.time()
    if n_cores == 1:
        xprint('Running %d simulations in 1/%d CPUs' % (n_sims, cpu_cores), group_log)
        for i in range(n_sims):
            sim_idx = i + 1
            if (sim_idx % 50 == 0) and (sim_idx > 1):
                xprint('\t clearing brian caches before sim %d...' % sim_idx, group_log)
                clear_brian_caches(clear_cython=True, clear_tex=True)

            options_single = choose_from_group_params(group_options, group_params, sim_idx, n_cores)
            run_single(options_single)

    # parse n_sims by n_cores
    elif n_cores > 1:
        xprint('Running %d simulations in %d/%d CPUs' % (n_sims, n_cores, cpu_cores), group_log)
        sim_idx = 1
        n_sims_left = n_sims

        runs_since_clear_cache = 0
        # clear_brian_caches(clear_cython=True, clear_tex=True)

        n_steps = (n_sims // n_cores) + 1
        for k in range(n_steps):
            if runs_since_clear_cache > 100:
                runs_since_clear_cache = 0
                xprint('\t clearing brian caches before sim %d...' % sim_idx, group_log)
                clear_brian_caches(clear_cython=True, clear_tex=True)

            if n_sims_left > 0:
                n_cores_step = min(n_cores, n_sims_left)

                xprint('\t running simulations %d-%d in %d CPUs' % (sim_idx, sim_idx + n_cores_step - 1,
                                                                    n_cores_step), group_log)

                start_step_time = time.time()

                settings_array = [None] * n_cores_step
                for i in range(n_cores_step):
                    settings_array[i] = choose_from_group_params(group_options, group_params, sim_idx, n_cores_step)
                    sim_idx += 1
                    runs_since_clear_cache += 1

                p = Pool(n_cores_step)
                p.map(run_single, settings_array)
                n_sims_left -= n_cores_step

                end_step_time = time.time() - start_step_time
                xprint('\t\t %s: finished step in %s. %d simulations left...' %
                       (time.strftime("%H:%M:%S"), seconds_to_hhmmss(end_step_time), n_sims_left), group_log)

    else:
        raise ValueError('Number of cores must be >= 1!')

    end_time = (time.time() - start_time)
    xprint('Finished simulation group %s in %s' % (group_options['time_stamp'], seconds_to_hhmmss(end_time)), group_log)


def initialize_sim(options, specified_params):
    """
    initializes an individual brian simulation
    """

    # import default parameters
    sim_params = get_dft_sim_params()
    plot_params = get_dft_plot_params()

    # log command line:
    log = (options['output_dir'] + options['time_stamp'] + '/' + 'sim' + str(options['sim_idx']) +
           '.' + options['group_param_array_str'] + '.log')

    # load specified parameters:
    load_param_array(sim_params, specified_params)

    # load from group param arrays:
    if 'group_param_array' in options:
        xprint('Simulation parameters:', log)
        load_param_array(sim_params, options['group_param_array'], log)

    # initialise python seed:
    sim_seed = sim_params['sim_seed'].get_param()
    random.seed(sim_seed)

    return log, sim_params, plot_params


def sort_events(events, monitors):
    """
    sorts simulation events and monitors, such that
    the brian simulation can run them in the intended order
    Args:
        events: list of simulation events
        monitors: list of simulation recording monitors
    """
    sim_dt = defaultclock.dt
    event_obj = []
    event_times = []
    event_types = []
    for event in (events + monitors):
        if isinstance(event, MonitorSettings):
            event_obj += [event]
            event_times += [event.start_record]
            event_types += ['monitor_on']

            event_obj += [event]
            event_times += [event.stop_record + sim_dt]
            event_types += ['monitor_off']

        elif isinstance(event, VoltageStimulus):
            jump_times = list(event.jump_dict.keys())
            event_obj += [event]
            event_times += [np.min(jump_times) * second]
            event_types += ['volt_jump_on']

            event_obj += [event]
            event_times += [np.max(jump_times) * second + sim_dt]
            event_types += ['volt_jump_off']

        elif isinstance(event, NetworkTests):
            event_obj += [event]
            event_times += [event.start]
            event_types += ['test_on']

            event_obj += [event]
            event_times += [event.stop]
            event_types += ['test_off']

        elif isinstance(event, ChangeAttribute):
            event_obj += [event]
            event_times += [event.onset]
            event_types += ['change_attribute']

    event_order = np.argsort(event_times / second)
    ordered_event_objs = [event_obj[i] for i in event_order]
    ordered_event_times = [event_times[i] for i in event_order]
    ordered_event_types = [event_types[i] for i in event_order]

    return ordered_event_objs, ordered_event_times, ordered_event_types


def run_network_sim(settings):
    """
    runs a brian network simulation
    """

    log = settings.log
    sim_params = settings.sim_params
    built_network = settings.brian_net
    net_objects = settings.net_objects
    events = settings.events
    monitors = settings.monitors

    # initialise Brian time step
    sim_dt = sim_params['sim_dt'].get_param()
    defaultclock.dt = sim_dt

    # initialise network with the specified initial conditions
    sim_seed = sim_params['sim_seed'].get_param()
    init_network(built_network, net_objects, sim_params)

    sim_idx = settings.options['sim_idx']
    xprint('\n====== Starting Simulation %d ======' % (sim_idx), log)

    start_time = time.time()

    # sort order of simulations events, tests, and plots
    ordered_event_objs, ordered_event_times, ordered_event_types = sort_events(events, monitors)

    """ BRIAN SIMULATION STARTS HERE """
    running_monitor_groups = []
    for i in range(len(ordered_event_objs)):
        time_to_next_event = ordered_event_times[i] - built_network.t

        if time_to_next_event / second > 0.1:
            built_network.run(time_to_next_event, report='text', report_period=60 * second)
        elif time_to_next_event / second > 0.:
            built_network.run(time_to_next_event, report_period=60 * second)

        event_obj = ordered_event_objs[i]

        if ordered_event_types[i] == 'volt_jump_on':
            event_obj.brian_pop = built_network['pop_' + event_obj.target.name]

            built_network.add(event_obj.operation)

        elif ordered_event_types[i] == 'volt_jump_off':
            built_network.remove(event_obj.operation)

        elif ordered_event_types[i] == 'monitor_on':
            event_obj.create_monitors(built_network, settings, log)
            running_monitor_groups += [event_obj]

        elif ordered_event_types[i] == 'test_off':
            test_range = [event_obj.start, event_obj.stop]

            test_data = {}
            # perform tests:
            xprint('\n========== TEST RESULTS %s ==================' % test_range, log)
            event_obj.perform_tests(built_network, settings, test_range, test_data, log)

            # create plots:
            start_plot_time = time.time()
            xprint('Creating plot for [%.3f-%.3f]s ...' % (test_range[0] / second, test_range[1] / second), log)
            event_obj.create_plots(built_network, settings, events, test_range, test_data)
            end_plot_time = time.time() - start_plot_time
            xprint('Created plot for [%.3f-%.3f]s in %.2f seconds' %
                   (test_range[0] / second, test_range[1] / second, end_plot_time), log)

        elif ordered_event_types[i] == 'monitor_off':
            event_obj.delete_monitors(built_network, running_monitor_groups, log)
            running_monitor_groups.remove(event_obj)

        elif ordered_event_types[i] == 'change_attribute':
            change_success = False
            if event_obj.target in built_network:
                if hasattr(built_network[event_obj.target], event_obj.attribute):
                    target_pop = built_network[event_obj.target]
                    if event_obj.subset > 0:
                        target_pop = target_pop[: event_obj.subset]

                    setattr(target_pop,
                            event_obj.attribute,
                            event_obj.value)

                    xprint('%.4f s: changed %s.%s[:%d] to %s' %
                           (built_network.t / second,
                            event_obj.target,
                            event_obj.attribute,
                            event_obj.subset,
                            event_obj.value), log)
                    change_success = True

            if not change_success:
                xprint("%.4f s: ERROR! Failed to change attribute '%s' on target '%s'" %
                       (built_network.t / second,
                        event_obj.attribute,
                        event_obj.target), log)

    """ BRIAN SIMULATION ENDED """

    test_dur = time.time() - start_time
    xprint('\n====== Finished Simulation %s ======' % (sim_idx), log)
    xprint('Simulation completed in %s\n' % seconds_to_hhmmss(test_dur), log)

    for pop_name in net_objects.pop_settings:
        pop = net_objects.pop_settings[pop_name]
        xprint('\n========== Population %s Settings ===========\n' % pop.name.upper(), log)
        xprint('type: %s' % type(pop), log)
        pop_print = print_attr(pop)
        xprint(pop_print, log)

    for syn_name in net_objects.syn_settings:
        syn = net_objects.syn_settings[syn_name]
        xprint('\n========== Synapse %s Settings ===========\n' % syn.name.upper(), log)
        xprint('type: %s' % type(syn), log)
        syn_print = print_attr(syn)
        xprint(syn_print, log)

    xprint('\n=========== Used Simulation Parameters ============\n', log)
    sim_params = settings.sim_params
    for p in sim_params:
        if sim_params[p].used:
            xprint('%s = %s' % (p, sim_params[p].get_param()), log)

    xprint('\n========== Finished Printing Results ===========\n', log)
