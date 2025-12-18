"""
simulations.py
This module provides functions and classes for running spiking network simulations using Brian2.
It handles simulation setup, event ordering, network execution, and logging of results and parameters.
"""

import time
import numpy as np
from brian2 import second, defaultclock
from general_code.parameters import print_attr
from general_code.aux_functions import xprint, seconds_to_hhmmss
from spiking_networks.tests import NetworkTests, MonitorSettings
from spiking_networks.network import init_network, TriggerSpikes, ChangeAttribute



class SimElement:
    """
    Container for simulation properties and objects.
    Stores all relevant settings, network objects, parameters, events, monitors, and log file.
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


def sort_events(events, monitors):
    """
    Sort simulation events and monitors so Brian2 can execute them in the intended chronological order.

    Args:
        events: List of simulation events (tests, spike triggers, attribute changes).
        monitors: List of simulation recording monitors.

    Returns:
        ordered_event_objs: List of event/monitor objects in sorted order.
        ordered_event_times: List of event times in sorted order.
        ordered_event_types: List of event types in sorted order.
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

        elif isinstance(event, TriggerSpikes):
            event_obj += [event]
            event_times += [event.time]
            event_types += ['stim_spikes']

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
    Run a Brian2 spiking network simulation with the specified settings and events.

    Args:
        settings: SimElement object containing all simulation configuration.

    Returns:
        None. Results and logs are written to disk.
    """

    log = settings.log
    sim_params = settings.sim_params
    built_network = settings.brian_net
    net_objects = settings.net_objects
    events = settings.events
    monitors = settings.monitors


    # Initialise Brian2 time step
    sim_dt = sim_params['sim_dt'].get_param()
    defaultclock.dt = sim_dt

    # Initialise network with the specified initial conditions
    sim_seed = sim_params['sim_seed'].get_param()
    init_network(built_network, net_objects, sim_params)

    sim_idx = settings.options['sim_idx']
    xprint('\n====== Starting Simulation %d ======' % (sim_idx), log)

    start_time = time.time()

    # Sort order of simulation events, tests, and plots
    ordered_event_objs, ordered_event_times, ordered_event_types = sort_events(events, monitors)

    # --- BRIAN2 SIMULATION STARTS HERE ---
    running_monitor_groups = []
    for i in range(len(ordered_event_objs)):
        time_to_next_event = ordered_event_times[i] - built_network.t

        # Run simulation up to the next event
        if time_to_next_event / second > 0.1:
            built_network.run(time_to_next_event, report='text', report_period=60 * second)
        elif time_to_next_event / second > 0.:
            built_network.run(time_to_next_event, report_period=60 * second)

        event_obj = ordered_event_objs[i]

        # Handle each event type
        if ordered_event_types[i] == 'stim_spikes':
            # Trigger spikes in the target population
            event_obj.brian_pop = built_network['pop_' + event_obj.target.name]
            event_obj.make_spikes()

        elif ordered_event_types[i] == 'monitor_on':
            # Start recording monitors
            event_obj.create_monitors(built_network, settings, log)
            running_monitor_groups += [event_obj]

        elif ordered_event_types[i] == 'test_off':
            # End a test, perform analysis and plotting
            test_range = [event_obj.start, event_obj.stop]

            test_data = {}
            xprint('\n========== TEST RESULTS %s ==================' % test_range, log)
            event_obj.perform_tests(built_network, settings, test_range, test_data, log)

            # Create plots for the test interval
            start_plot_time = time.time()
            xprint('Creating plot for [%.3f-%.3f]s ...' % (test_range[0] / second, test_range[1] / second), log)
            event_obj.create_plots(built_network, settings, test_range)
            end_plot_time = time.time() - start_plot_time
            xprint('Created plot for [%.3f-%.3f]s in %.2f seconds' %
                   (test_range[0] / second, test_range[1] / second, end_plot_time), log)

        elif ordered_event_types[i] == 'monitor_off':
            # Stop recording monitors
            event_obj.delete_monitors(built_network, running_monitor_groups, log)
            running_monitor_groups.remove(event_obj)

        elif ordered_event_types[i] == 'change_attribute':
            # Change an attribute of a network object (e.g., parameter update)
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
                xprint("%.4f s: WARNING! Failed to change attribute '%s' on target '%s'" %
                       (built_network.t / second,
                        event_obj.attribute,
                        event_obj.target), log)

    # --- BRIAN2 SIMULATION ENDED ---

    test_dur = time.time() - start_time
    xprint('\n====== Finished Simulation %s ======' % (sim_idx), log)
    xprint('Simulation completed in %s\n' % seconds_to_hhmmss(test_dur), log)

    # Print population settings
    for pop_name in net_objects.pop_settings:
        pop = net_objects.pop_settings[pop_name]
        xprint('\n========== Population %s Settings ===========\n' % pop.name.upper(), log)
        xprint('type: %s' % type(pop), log)
        pop_print = print_attr(pop)
        xprint(pop_print, log)

    # Print synapse settings
    for syn_name in net_objects.syn_settings:
        syn = net_objects.syn_settings[syn_name]
        xprint('\n========== Synapse %s Settings ===========\n' % syn.name.upper(), log)
        xprint('type: %s' % type(syn), log)
        syn_print = print_attr(syn)
        xprint(syn_print, log)

    # Print used simulation parameters
    xprint('\n=========== Used Simulation Parameters ============\n', log)
    sim_params = settings.sim_params
    for p in sim_params:
        if sim_params[p].used:
            xprint('%s = %s' % (p, sim_params[p].get_param()), log)

    xprint('\n========== Finished Printing Results ===========\n', log)
