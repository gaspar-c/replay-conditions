"""
slurm_worker.py
Entry point for individual Slurm array jobs submitted by run_sim_group.

Called automatically via sbatch — do not run directly.
Usage (internal): python -m general_code.slurm_worker <params_pkl> <module_path> <func_name>
"""
import os
import sys
import pickle
import shutil
import fcntl
import importlib


def _set_brian2_cache_dir():
    """Copy the pre-compiled NFS cache to a node-local /tmp dir (once per node),
    then point Brian2 there to avoid NFS contention under concurrent array tasks."""
    import brian2
    nfs_cache  = os.environ.get('BRIAN2_CACHE_DIR')
    array_id   = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'default'))
    tmpdir     = os.environ.get('TMPDIR', '/tmp')
    local_cache = os.path.join(tmpdir, f'brian_cache_{array_id}')
    lock_path   = local_cache + '.lock'

    if nfs_cache:
        os.makedirs(nfs_cache, exist_ok=True)
        if os.listdir(nfs_cache):
            # NFS cache already populated (warmup done) — copy to node-local /tmp.
            # Serialise concurrent arrivals; only the first task on this node copies.
            with open(lock_path, 'w') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                if not os.path.isdir(local_cache):
                    shutil.copytree(nfs_cache, local_cache)
                fcntl.flock(lf, fcntl.LOCK_UN)
            cache_dir = local_cache
        else:
            # NFS cache dir exists but is empty — this is the warmup job; compile there.
            cache_dir = nfs_cache
    else:
        # No NFS cache configured — compile to a node-local directory.
        os.makedirs(local_cache, exist_ok=True)
        cache_dir = local_cache

    brian2.prefs.codegen.runtime.cython.cache_dir = cache_dir


def main():
    if len(sys.argv) < 4:
        print("Usage: python -m general_code.slurm_worker <params_pkl> <module_path> <func_name>")
        sys.exit(1)

    params_pkl  = sys.argv[1]
    module_path = sys.argv[2]
    func_name   = sys.argv[3]

    _set_brian2_cache_dir()

    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_task_id is None:
        print("Error: SLURM_ARRAY_TASK_ID not set. This script must be run as a Slurm array job.")
        sys.exit(1)
    sim_idx = int(slurm_task_id)

    with open(params_pkl, 'rb') as f:
        data = pickle.load(f)
    group_options = data['group_options']
    group_params  = data['group_params']

    module     = importlib.import_module(module_path)
    run_single = getattr(module, func_name)

    from general_code.group_simulations import choose_from_group_params
    options_single = choose_from_group_params(group_options, group_params, sim_idx, n_cores=1)
    run_single(options_single)


if __name__ == '__main__':
    main()
