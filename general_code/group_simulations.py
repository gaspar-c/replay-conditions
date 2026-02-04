"""
group_simulations.py
Utility functions for running and managing groups of spiking network simulations.
Handles parameter selection, parallel execution, and logging for batch simulation workflows.
"""
import os
import time
import socket
import sys
import multiprocessing as mltp
from brian2 import clear_cache
from general_code.aux_functions import xprint, seconds_to_hhmmss


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


def choose_from_group_params(settings, group_params, sim_idx, n_cores):
    """
    Select parameters for a specific simulation in a group.

    Args:
        settings: Simulation settings.
        group_params: Parameters for group of simulations.
        sim_idx: Simulation index (within group).
        n_cores: Number of CPU cores.

    Returns:
        dict: Simulation settings for the selected simulation.
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
    sim_settings['group_param_overrides'] = select_params
    sim_settings['group_param_array_str'] = param_array_str(select_params)

    return sim_settings


def run_sim_group(group_options, group_params, run_single):
    """
    run group of simulations. This function distributes the specified
    settings and parameters for the whole simulation group to each individual
    simulation, running each of them in parallel (if several CPUs are available)
    """

    """ CREATE SIMULATION GROUP OUTPUT FOLDER """
    group_log = group_options['output_dir'] + group_options['group_label'] + '/0_group_log.log'
    os.makedirs(group_options['output_dir'] + group_options['group_label'], exist_ok=True)

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
    cpu_cores = mltp.cpu_count()
    if 'max_cores' in group_options:
        max_cores = min([group_options['max_cores'], cpu_cores])
    else:
        max_cores = cpu_cores
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
                xprint('\t clearing caches before sim %d...' % sim_idx, group_log)
                try:
                    clear_cache('cython')
                except FileNotFoundError:
                    pass  # Ignore if the cache directory does not exist

            options_single = choose_from_group_params(group_options, group_params, sim_idx, n_cores)
            run_single(options_single)

    # parse n_sims by n_cores
    elif n_cores > 1:
        xprint('Running %d simulations in %d/%d CPUs' % (n_sims, n_cores, cpu_cores), group_log)
        sim_idx = 1
        n_sims_left = n_sims

        runs_since_clear_cache = 0

        n_steps = (n_sims // n_cores) + 1
        for k in range(n_steps):
            if runs_since_clear_cache > 100:
                runs_since_clear_cache = 0
                xprint('\t clearing caches before sim %d...' % sim_idx, group_log)
                try:
                    clear_cache('cython')
                except FileNotFoundError:
                    pass  # Ignore if the cache directory does not exist

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

                p = mltp.Pool(n_cores_step)
                p.map(run_single, settings_array)
                n_sims_left -= n_cores_step

                end_step_time = time.time() - start_step_time
                xprint('\t\t %s: finished step in %s. %d simulations left...' %
                       (time.strftime("%H:%M:%S"), seconds_to_hhmmss(end_step_time), n_sims_left), group_log)

    else:
        raise ValueError('Number of cores must be >= 1!')

    end_time = (time.time() - start_time)
    xprint('Finished simulation group %s in %s' % (group_options['group_label'], seconds_to_hhmmss(end_time)), group_log)

