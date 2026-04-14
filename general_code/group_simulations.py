"""
group_simulations.py
Utility functions for running and managing groups of spiking network simulations.
Handles parameter selection, parallel execution, and logging for batch simulation workflows.
"""
import os
import time
import socket
import sys
import shutil
import pickle
import subprocess
import multiprocessing as mltp
from general_code.aux_functions import xprint, seconds_to_hhmmss


_SLURM_DEFAULTS = {
    'partition': 'short',
    'mem': '8G',
    'cpus_per_task': 1,
}


def _warmup_sim_idx(group_params, n_sims):
    """Return the index of the first sim where all numeric params are non-zero.

    Using the first non-zero sim (rather than sim 1 which may have 0 connectivity)
    ensures Brian2 compiles synapse-related Cython modules into the shared cache.
    Falls back to n_sims if every sim has at least one zero numeric param.
    """
    for idx in range(1, n_sims + 1):
        values = []
        for param_array in group_params.values():
            val = param_array[idx - 1] if idx - 1 < len(param_array) else param_array[-1]
            if isinstance(val, (int, float)):
                values.append(val)
        if values and all(v != 0 for v in values):
            return idx
    return n_sims


def _submit_slurm_array(group_options, group_params, run_single, n_sims, group_log):
    """Serialize params, write sbatch scripts, and submit a Slurm job array.

    Submission is two-step to avoid per-job Brian2 Cython recompilation:
      1. A warmup job (sim 1) compiles Brian2 into a shared cache dir and runs sim 1.
      2. The main array (sims 2..n_sims) starts after the warmup completes, with
         BRIAN2_CACHE_DIR pointing at the pre-compiled cache.
    """
    out_dir   = group_options['output_dir'] + group_options['group_label']
    cache_dir = os.path.join(out_dir, 'brian_cache')

    # Serialize params for workers
    pkl_path = os.path.join(out_dir, 'slurm_params.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'group_options': group_options, 'group_params': group_params}, f)

    # Separate pickle for the warmup worker: same params but with compile_only=True
    warmup_options = dict(group_options)
    warmup_options['compile_only'] = True
    warmup_pkl_path = os.path.join(out_dir, 'slurm_params_warmup.pkl')
    with open(warmup_pkl_path, 'wb') as f:
        pickle.dump({'group_options': warmup_options, 'group_params': group_params}, f)

    # Slurm options: defaults, overridden by group_options['slurm'] if present
    slurm_opts = dict(_SLURM_DEFAULTS)
    slurm_opts.update(group_options.get('slurm', {}))

    module_path = run_single.__module__
    func_name   = run_single.__name__
    python_exe  = sys.executable
    work_dir    = os.getcwd()
    log_dir     = out_dir

    time_line     = f"#SBATCH --time={slurm_opts['time']}\n" if 'time' in slurm_opts else ""
    slurm_header  = (
        f"#SBATCH --partition={slurm_opts['partition']}\n"
        + time_line
        + f"#SBATCH --mem={slurm_opts['mem']}\n"
        f"#SBATCH --cpus-per-task={slurm_opts['cpus_per_task']}\n"
    )
    worker_cmd        = f"{python_exe} -m general_code.slurm_worker {pkl_path} {module_path} {func_name}"
    warmup_worker_cmd = f"{python_exe} -m general_code.slurm_worker {warmup_pkl_path} {module_path} {func_name}"

    # --- warmup script: compile-only run (t=0) to populate the shared Brian2 cache.
    #     Exits in seconds so all n_sims array jobs can start in parallel right after. ---
    warmup_idx = _warmup_sim_idx(group_params, n_sims)
    warmup_script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={group_options['group_label']}_warmup\n"
        + slurm_header
        + f"#SBATCH --output={log_dir}/slurm_warmup.log\n"
        "\n"
        f"export BRIAN2_CACHE_DIR={cache_dir}\n"
        f"export SLURM_ARRAY_TASK_ID={warmup_idx}\n"
        f"cd {work_dir}\n"
        f"{warmup_worker_cmd}\n"
    )
    warmup_path = os.path.join(out_dir, 'job_warmup.sh')
    with open(warmup_path, 'w') as f:
        f.write(warmup_script)

    warmup_result = subprocess.run(['sbatch', warmup_path], capture_output=True, text=True)
    if warmup_result.returncode != 0:
        xprint(f'sbatch warmup failed: {warmup_result.stderr}', group_log)
        raise RuntimeError(f'sbatch warmup submission failed:\n{warmup_result.stderr}')
    warmup_job_id = warmup_result.stdout.strip().split()[-1]
    xprint(f'Submitted warmup job {warmup_job_id} (sim {warmup_idx}, first non-zero params). Script: {warmup_path}', group_log)

    # --- main array: all n_sims, starts only after warmup (compile-only) succeeds ---
    array_script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={group_options['group_label']}\n"
        + slurm_header
        + f"#SBATCH --array=1-{n_sims}\n"
        f"#SBATCH --output={log_dir}/slurm_%A_%a.log\n"
        f"#SBATCH --dependency=afterok:{warmup_job_id}\n"
        "\n"
        f"export BRIAN2_CACHE_DIR={cache_dir}\n"
        f"cd {work_dir}\n"
        f"{worker_cmd}\n"
    )
    array_path = os.path.join(out_dir, 'job.sh')
    with open(array_path, 'w') as f:
        f.write(array_script)

    array_result = subprocess.run(['sbatch', array_path], capture_output=True, text=True)
    if array_result.returncode == 0:
        array_job_id = array_result.stdout.strip()
        xprint(f'Submitted {n_sims} array jobs to Slurm ({array_job_id}), '
               f'pending warmup {warmup_job_id}. Script: {array_path}', group_log)
    else:
        xprint(f'sbatch array failed: {array_result.stderr}', group_log)
        raise RuntimeError(f'sbatch array submission failed:\n{array_result.stderr}')


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
    Run a group of simulations.

    On systems with Slurm (sbatch available), submits a job array and returns
    True immediately — simulations run asynchronously on the cluster.

    On systems without Slurm, runs simulations locally in parallel using
    multiprocessing and returns False.

    To customise Slurm options (partition, time limit, memory), add a 'slurm'
    key to group_options, e.g.:
        group_options['slurm'] = {'partition': 'normal', 'time': '08:00:00', 'mem': '16G'}
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

    """ USE SLURM IF AVAILABLE """
    if shutil.which('sbatch') is not None:
        _submit_slurm_array(group_options, group_params, run_single, n_sims, group_log)
        return

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

    # Run warmup sim first to populate the Brian2 Cython cache. All subsequent
    # workers (serial or parallel) will read the pre-compiled cache instead of
    # compiling from scratch, avoiding lock contention and stuck processes.
    warmup_idx = _warmup_sim_idx(group_params, n_sims)
    warmup_options = dict(group_options)
    warmup_options['compile_only'] = True
    xprint('Running compile-only warmup (sim %d) to populate Brian2 cache...' % warmup_idx, group_log)
    run_single(choose_from_group_params(warmup_options, group_params, warmup_idx, n_cores))
    all_idxs = list(range(1, n_sims + 1))

    if n_cores == 1:
        xprint('Running %d simulations in 1/%d CPUs' % (n_sims, cpu_cores), group_log)
        for sim_idx in all_idxs:
            options_single = choose_from_group_params(group_options, group_params, sim_idx, n_cores)
            run_single(options_single)

    # parse n_sims by n_cores
    elif n_cores > 1:
        xprint('Running %d simulations in %d/%d CPUs' % (n_sims, n_cores, cpu_cores), group_log)
        offset = 0

        n_steps = (n_sims // n_cores) + 1
        for k in range(n_steps):
            n_sims_left = n_sims - offset
            if n_sims_left > 0:
                n_cores_step = min(n_cores, n_sims_left)
                batch = all_idxs[offset:offset + n_cores_step]

                xprint('\t running simulations %d-%d in %d CPUs' % (batch[0], batch[-1], n_cores_step), group_log)

                start_step_time = time.time()

                settings_array = [choose_from_group_params(group_options, group_params, sim_idx, n_cores_step)
                                  for sim_idx in batch]
                offset += n_cores_step

                p = mltp.Pool(n_cores_step)
                p.map(run_single, settings_array)

                end_step_time = time.time() - start_step_time
                xprint('\t\t %s: finished step in %s. %d simulations left...' %
                       (time.strftime("%H:%M:%S"), seconds_to_hhmmss(end_step_time), n_sims - offset), group_log)

    else:
        raise ValueError('Number of cores must be >= 1!')

    end_time = (time.time() - start_time)
    xprint('Finished simulation group %s in %s' % (group_options['group_label'], seconds_to_hhmmss(end_time)), group_log)

